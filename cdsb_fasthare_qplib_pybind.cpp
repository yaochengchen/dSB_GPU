// main.cpp
#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "fasthare_api.h"

// =========================
// pybind11 embed (call Python CDSB)
// =========================
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// =========================
// Dense aliases (standard mode only)
// =========================
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

// =========================
// Enable calling Python CDSB
// Default ON for validation. You can disable with -DUSE_PY_CDSB=0
// =========================
#ifndef USE_PY_CDSB
#define USE_PY_CDSB 1
#endif

// =========================
// Where to find python module (optional)
// You can override at compile time, e.g. -DPY_CDSB_SYSPATH=\"../my_dSB_project\"
// =========================
#ifndef PY_CDSB_SYSPATH
#define PY_CDSB_SYSPATH "./"
#endif

// =========================
// Python module + class names (override if yours differ)
// =========================
#ifndef PY_CDSB_MODULE
#define PY_CDSB_MODULE "dSB_original_python"
#endif

#ifndef PY_CDSB_CLASS
#define PY_CDSB_CLASS "CDSB"
#endif

// ========================================================================
// to_standard_form (dense -> sk_ising standard form)
// J is vector of (u, v, w) where u<v represent couplers, and bias terms are
// encoded as (i, n, h_i) with extra bias node index n.
// negate logic kept consistent with your original: if !negate => weight = -val
// ========================================================================
static inline fasthare_api::ski to_standard_form_dense_upper(
    const Mat& H,
    const Vec& h,
    bool negate
) {
    const int n = static_cast<int>(h.size());
    if (H.rows() != n || H.cols() != n) {
        throw std::runtime_error("to_standard_form_dense_upper: H shape mismatch with h");
    }

    fasthare_api::ski J;
    J.reserve(static_cast<size_t>(n) * static_cast<size_t>(n - 1) / 2 + static_cast<size_t>(n));

    // upper triangle only (i<j)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double val = H(i, j);
            if (val == 0.0) continue;
            const double w = (!negate) ? -val : val;
            J.emplace_back(i, j, w);
        }
    }

    // bias: (i, n, h_i)
    for (int i = 0; i < n; ++i) {
        const double hi = h(i);
        if (hi == 0.0) continue;
        const double hp = (!negate) ? -hi : hi;
        J.emplace_back(i, n, hp);
    }
    return J;
}

// ========================================================================
// FastHare reduction wrapper (dense in/out)
// ========================================================================
struct FHResult {
    bool Flag = false;            // true => 100% reduction (Jr empty)
    std::vector<int> sign;
    Mat Hr;                       // reduced dense matrix (when Flag=false)
    std::vector<int> spin_map;    // map back to original spins (when Flag=false)
    double runtime = 0.0;
};

static inline FHResult fh_for_ising_dense(
    const Mat& H,
    const Vec& h,
    bool in_negate,
    double alpha
) {
    fasthare_api::ski J = to_standard_form_dense_upper(H, h, in_negate);

    // C++ API: tuple<ski Jr, vector<int> spin_map, vector<int> sign, double runtime>
    auto out = fasthare_api::fasthare_reduction(J, /*file=*/"", alpha, /*log_file=*/"");
    const auto& Jr       = std::get<0>(out);
    const auto& spin_map = std::get<1>(out);
    const auto& sign     = std::get<2>(out);
    const double runtime = std::get<3>(out);

    FHResult fr;
    fr.sign    = sign;
    fr.runtime = runtime;
    fr.Flag    = Jr.empty();

    if (fr.Flag) {
        // 100% reduction: no Hr/spin_map needed
        return fr;
    }

    fr.spin_map = spin_map;

    // Determine reduced size m from spin_map (max index + 1)
    int m = 0;
    for (int s : fr.spin_map) m = std::max(m, s);
    m += 1;

    // Build A(u,v) = -w then symmetrize: Hr = A + A^T
    Mat A = Mat::Zero(m, m);
    for (const auto& e : Jr) {
        int u, v;
        double w;
        std::tie(u, v, w) = e;
        if (0 <= u && u < m && 0 <= v && v < m) {
            A(u, v) = -w;
        }
    }
    fr.Hr = A + A.transpose();
    return fr;
}

// ========================================================================
// QPLIB translate (dense)
// Returns:
// - input_J, input_h : scaled dense
// - J, h             : transformed dense (unscaled)
// - offset
// Logic mirrors your original code path.
// ========================================================================
struct QplibResult {
    Mat input_J;
    Vec input_h;
    Mat J;
    Vec h;
    double offset = 0.0;
};

static inline QplibResult qplib_translate_dense(const std::string& data_name) {
    std::ifstream fin(data_name);
    if (!fin) throw std::runtime_error("Cannot open file: " + data_name);

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(fin, line)) lines.push_back(line);
    if (lines.size() < 8) throw std::runtime_error("QPLIB file too short: " + data_name);

    int Edge = 0;
    {
        std::istringstream iss(lines.at(4));
        iss >> Edge;
    }
    if (Edge < 0) throw std::runtime_error("Invalid Edge count in file: " + data_name);

    struct EdgeRec { int a; int b; double w; };
    std::vector<EdgeRec> Jdata;
    Jdata.reserve(static_cast<size_t>(Edge));

    for (int i = 5; i < 5 + Edge; ++i) {
        if (i >= (int)lines.size()) throw std::runtime_error("Unexpected EOF while reading edges: " + data_name);
        std::istringstream eiss(lines.at(i));
        EdgeRec r{};
        eiss >> r.a >> r.b >> r.w;
        Jdata.push_back(r);
    }

    double hvalue = 0.0;
    {
        const int idx = 5 + Edge;
        if (idx >= (int)lines.size()) throw std::runtime_error("Unexpected EOF while reading hvalue: " + data_name);
        std::istringstream hiss(lines.at(idx));
        hiss >> hvalue;
    }

    int hnum = 0;
    {
        const int idx = 5 + Edge + 1;
        if (idx >= (int)lines.size()) throw std::runtime_error("Unexpected EOF while reading hnum: " + data_name);
        std::istringstream nss(lines.at(idx));
        nss >> hnum;
    }
    if (hnum < 0) throw std::runtime_error("Invalid hnum in file: " + data_name);

    struct HRec { int idx; double val; };
    std::vector<HRec> hdata;
    hdata.reserve(static_cast<size_t>(hnum));

    for (int i = 7 + Edge; i < 7 + Edge + hnum; ++i) {
        if (i >= (int)lines.size()) throw std::runtime_error("Unexpected EOF while reading h entries: " + data_name);
        std::istringstream iss2(lines.at(i));
        HRec r{};
        iss2 >> r.idx >> r.val;
        hdata.push_back(r);
    }

    // Build maps (1-based indices in QPLIB)
    auto pack2 = [](int a, int b) -> long long {
        return (static_cast<long long>(a) << 32) ^ static_cast<unsigned int>(b);
    };

    std::unordered_map<long long, double> J_info;
    J_info.reserve(Jdata.size() * 2);
    for (const auto& j : Jdata) {
        // your original: w/2
        J_info[pack2(j.a, j.b)] = j.w / 2.0;
    }

    std::unordered_map<int, double> h_info;
    h_info.reserve(hdata.size() * 2);
    for (const auto& hi : hdata) {
        // your original: idx-1
        h_info[hi.idx - 1] = hi.val;
    }

    // n_vertex = max edge endpoint (still 1-based), then used as size
    int n_vertex = 0;
    for (const auto& kv : J_info) {
        const long long key = kv.first;
        const int a = static_cast<int>(key >> 32);
        const int b = static_cast<int>(static_cast<unsigned int>(key));
        n_vertex = std::max(n_vertex, std::max(a, b));
    }
    if (n_vertex <= 0) throw std::runtime_error("n_vertex <= 0 parsed from file: " + data_name);

    // Dense symmetric J (0-based in matrix)
    Mat J = Mat::Zero(n_vertex, n_vertex);
    for (const auto& kv : J_info) {
        const long long key = kv.first;
        const int a1 = static_cast<int>(key >> 32);
        const int b1 = static_cast<int>(static_cast<unsigned int>(key));
        const int a = a1 - 1;
        const int b = b1 - 1;
        const double w = kv.second;
        if (0 <= a && a < n_vertex && 0 <= b && b < n_vertex) {
            J(a, b) = w;
            J(b, a) = w;
        }
    }

    // Dense h
    Vec h = Vec::Zero(n_vertex);
    if (!h_info.empty()) {
        h.array() = hvalue;
        for (const auto& kv : h_info) {
            const int idx = kv.first;
            if (0 <= idx && idx < n_vertex) h(idx) = kv.second;
        }
    }

    // Mirror your transformation:
    // J = J/8
    // offset = -J.sum() - h.sum()/2
    // h = h/2 + 2*J.sum(axis=1)
    // J = -2*J
    // h = -h
    J *= (1.0 / 8.0);

    const double Jsum = J.sum();
    const double hsum = h.sum();
    const double offset = -Jsum - hsum / 2.0;

    const Vec rowSum = J.rowwise().sum();
    h = h * 0.5 + 2.0 * rowSum;

    J *= (-2.0);
    h = -h;

    // scale_factor = max(abs(J).max(), abs(h).max())
    const double Jabsmax = J.cwiseAbs().maxCoeff();
    const double habsmax = h.cwiseAbs().maxCoeff();
    double scale_factor = std::max(Jabsmax, habsmax);
    if (scale_factor == 0.0) scale_factor = 1.0;

    Mat input_J = -J / scale_factor;
    Vec input_h = -h / scale_factor;

    QplibResult qr;
    qr.input_J = std::move(input_J);
    qr.input_h = std::move(input_h);
    qr.J       = std::move(J);
    qr.h       = std::move(h);
    qr.offset  = offset;
    return qr;
}

// ========================================================================
// Helpers: Eigen <-> numpy
// ========================================================================
static inline py::array_t<double> eigen_to_numpy_copy(const Mat& M) {
    py::array_t<double> arr({(py::ssize_t)M.rows(), (py::ssize_t)M.cols()});
    auto buf = arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < (py::ssize_t)M.rows(); ++i) {
        for (py::ssize_t j = 0; j < (py::ssize_t)M.cols(); ++j) {
            buf(i, j) = M((int)i, (int)j);
        }
    }
    return arr;
}

// ========================================================================
// Call Python CDSB:
//   from dSB_original_python import CDSB
//   dsb = CDSB(J=Hr, batch_size=..., n_iter=...)
//   dsb.update()
//   energy = dsb.calc_energy()
//   x = dsb.x
// Returns: xred (sign vector of best sample, values in {-1,0,1})
// ========================================================================
static inline std::vector<double> solve_reduced_with_python_cdsb(
    const Mat& Hr,
    int batch_size,
    int n_iter
) {
    // Ensure python can find your module
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("append")(PY_CDSB_SYSPATH);

    py::module_ mod = py::module_::import(PY_CDSB_MODULE);
    py::object CDSB_py = mod.attr(PY_CDSB_CLASS);

    // import torch
    py::module_ torch = py::module_::import("torch");

    // Python code uses CDSB(J=-hr, ...)
    Mat Hr_neg = -Hr;
    py::array_t<double> Hr_np = eigen_to_numpy_copy(Hr_neg);

    // numpy -> torch.tensor(float32)
    py::object Hr_torch = torch.attr("tensor")(Hr_np, py::arg("dtype") = torch.attr("float32"));

    py::object dsb = CDSB_py(
        py::arg("J") = Hr_torch,
        py::arg("batch_size") = batch_size,
        py::arg("n_iter") = n_iter
    );

    dsb.attr("update")();

    // ---- energy ----
    py::object e_obj = dsb.attr("calc_energy")();
    if (py::hasattr(e_obj, "detach")) {
        e_obj = e_obj.attr("detach")().attr("cpu")().attr("numpy")();
    } else if (!py::isinstance<py::array>(e_obj)) {
        // could be list; convert via numpy.array
        py::module_ np = py::module_::import("numpy");
        e_obj = np.attr("array")(e_obj);
    }

    py::array_t<float> e_np = e_obj.cast<py::array_t<float>>();
    if (e_np.ndim() != 1) throw std::runtime_error("energy is not 1D");

    auto eb = e_np.unchecked<1>();
    std::vector<double> energy((size_t)eb.shape(0));
    for (py::ssize_t i = 0; i < eb.shape(0); ++i) energy[(size_t)i] = (double)eb(i);

    int best = (int)(std::min_element(energy.begin(), energy.end()) - energy.begin());

    // ---- x ----
    py::object x_obj = dsb.attr("x");
    if (py::hasattr(x_obj, "detach")) {
        x_obj = x_obj.attr("detach")().attr("cpu")().attr("numpy")();
    } else if (!py::isinstance<py::array>(x_obj)) {
        py::module_ np = py::module_::import("numpy");
        x_obj = np.attr("array")(x_obj);
    }

    py::array_t<float> x_np = x_obj.cast<py::array_t<float>>();
    if (x_np.ndim() != 2) throw std::runtime_error("dsb.x is not 2D");

    auto xb = x_np.unchecked<2>();
    const int n_vars = (int)xb.shape(0);
    const int n_samp = (int)xb.shape(1);
    if (best < 0 || best >= n_samp) throw std::runtime_error("best index out of range for dsb.x");

    // xred = sign(x[:, best])
    std::vector<double> xred(n_vars, 0.0);
    for (int i = 0; i < n_vars; ++i) {
        const float v = xb(i, best);
        xred[i] = (v > 0.0f) ? 1.0 : (v < 0.0f ? -1.0 : 0.0);
    }
    return xred;
}


// ========================================================================
// main
// ========================================================================
int main(int argc, char** argv) {
    // alpha default 0.2; override: ./app 0.15
    double alpha = 0.2;
    if (argc >= 2) alpha = std::stod(argv[1]);

#if USE_PY_CDSB
    // Initialize Python interpreter ONCE for the whole program
    py::scoped_interpreter guard{};
#endif

    std::vector<int> qpnames = {
        3506, 3565, 3642, 3650, 3693, 3705, 3706, 3738,
        3745, 3822, 3832, 3838, 3850, 3852, 3877,
        5721, 5725, 5755, 5875
    };

    for (int qpname : qpnames) {
        std::cout << "==== QPLIB_" << qpname << " ====\n";
        const std::string data_name = "./data/QPLIB_" + std::to_string(qpname) + ".qplib";

        const double t1 = static_cast<double>(std::clock()) / CLOCKS_PER_SEC;

        QplibResult qr = qplib_translate_dense(data_name);

        FHResult fr = fh_for_ising_dense(
            -qr.input_J,
            -qr.input_h,
            /*in_negate=*/false,
            /*alpha=*/alpha
        );

        std::vector<double> solution;

        if (fr.Flag) {
            std::cout << "100% reduction\n";
            const int n = static_cast<int>(fr.sign.size());
            solution.resize(n);

            const int last = fr.sign.empty() ? 1 : fr.sign.back();
            for (int i = 0; i < n; ++i) {
                solution[i] = static_cast<double>(last * fr.sign[i]);
            }

            // If you want, you can compute energy here too (optional)
            Vec sol = Vec::Zero(qr.J.rows());
            for (int i = 0; i < sol.size() && i < (int)solution.size(); ++i) sol(i) = solution[i];

            const Vec Js = qr.J * sol;
            const double term1 = (Js.array() * sol.array()).sum() / 2.0;
            const double term2 = qr.h.dot(sol);
            const double object_energy = -(term1 + term2) - qr.offset;
            std::cout << "object_energy: " << object_energy << "\n";
        } else {
#if USE_PY_CDSB
            // ----- Python CDSB path (replace C++ CDSB) -----
            // Equivalent to:
            //   dsb = CDSB(J=-Hr, batch_size=1500, n_iter=8000)
            //   dsb.update()
            //   energy = dsb.calc_energy()
            //   best = argmin(energy)
            //   xred = sign(dsb.x[:,best])
            std::vector<double> xred = solve_reduced_with_python_cdsb(fr.Hr, /*batch_size=*/1500, /*n_iter=*/8000);

            // solution = xred[spin_map[i]] * sign[i]
            solution.resize(fr.spin_map.size());
            for (size_t i = 0; i < fr.spin_map.size(); ++i) {
                const int spin = fr.spin_map[i];
                double v = (0 <= spin && spin < (int)xred.size()) ? xred[spin] : 0.0;
                v *= static_cast<double>(fr.sign[i]);
                solution[i] = v;
            }

            // object_energy = -((J*sol)*sol/2 + h^T sol) - offset
            Vec sol = Vec::Zero(qr.J.rows());
            for (int i = 0; i < sol.size() && i < (int)solution.size(); ++i) sol(i) = solution[i];

            const Vec Js = qr.J * sol;
            const double term1 = (Js.array() * sol.array()).sum() / 2.0;
            const double term2 = qr.h.dot(sol);
            const double object_energy = -(term1 + term2) - qr.offset;
            std::cout << "object_energy: " << object_energy << "\n";
#else
            std::cout << "[Python CDSB disabled] Only reduction finished.\n";
#endif
        }

        const double t2 = static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
        std::cout << "use time: " << (t2 - t1) << "s\n\n";
    }

    return 0;
}

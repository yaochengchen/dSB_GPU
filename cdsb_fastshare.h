// cdsb_fastshare.h
#pragma once
// Pure C++ / CUDA CDSB implementation (no torch, no pybind)
// - Dense J on GPU stored as FP16 (__half)
// - x/y on GPU stored as FP16 (__half), layout: [N * B] with index (row * B + batch)
// - Update kernel: one block per batch, warp-per-row, shared memory staging in FP32.

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>    // NAN, std::isfinite

#ifndef CDSB_CUDA_CHECK
#define CDSB_CUDA_CHECK(call) do {                                       \
  cudaError_t _e = (call);                                               \
  if (_e != cudaSuccess) {                                               \
    throw std::runtime_error(std::string("CUDA error: ") +               \
      cudaGetErrorString(_e) + " @ " + __FILE__ + ":" + std::to_string(__LINE__)); \
  }                                                                      \
} while(0)
#endif

// -------- Public API --------
class CDSB {
public:
  using Mat = Eigen::MatrixXd; // host-facing

  // Python-aligned defaults:
  //   delta = 1
  //   dt    = 1
  //   xi    = auto: 0.5*sqrt(N-1)/sqrt(sum(J^2))   (if xi is NAN or 0)
  CDSB(const Mat& J, int batch_size, int n_iter,
       float delta = 1.0f, float xi = NAN, float dt = 1.0f);

  ~CDSB();

  // Run the CDSB dynamics on GPU; after update(), x is downloaded to host (MatrixXd)
  void update();

  // Return per-batch energy computed from current x (host x is used; will upload internally if needed)
  std::vector<double> calc_energy() const;

  // Host result: shape (N, B)
  Mat x;

private:
  int N_{0};
  int B_{0};
  int iters_{0};
  float delta_{1.0f};
  float xi_{NAN};
  float dt_{1.0f};

  // GPU buffers (FP16)
  __half* dJ_{nullptr};  // [N*N]
  __half* dx_{nullptr};  // [N*B]
  __half* dy_{nullptr};  // [N*B]
  __half* dp_{nullptr};  // [iters] p schedule in FP16

  // Host-side cached packed buffers (row-major) for upload
  std::vector<__half> hJ_; // [N*N]
  std::vector<__half> hx_; // [N*B]
  std::vector<__half> hy_; // [N*B]
  std::vector<__half> hp_; // [iters]

  mutable bool gpu_state_dirty_{false}; // host x changed since last upload

  void alloc_device_();
  void free_device_() noexcept;

  void init_random_xy_();    // Python-aligned init: Uniform(-0.01, 0.01)
  void build_p_schedule_();  // p[i] = i/(iters-1)

  void pack_J_to_half_(const Mat& J);
  void pack_x_to_half_();
  void unpack_x_from_half_();

  void upload_all_();
  void upload_x_if_dirty_() const;

  void auto_set_xi_from_J_(); // Python-aligned xi schedule if xi_ not provided

  // no copy
  CDSB(const CDSB&) = delete;
  CDSB& operator=(const CDSB&) = delete;
};

// Launchers implemented in .cu
void cdsb_fused_run_fp16(
    __half* dY,              // __half*
    __half* dX,              // __half*
    const __half* dJ,        // __half*
    const __half* dP,        // __half*
    float delta, float xi, float dt,
    int N, int B, int iters,
    cudaStream_t stream);

void cdsb_energy_fp16(
    const __half* dX,        // __half*
    const __half* dJ,        // __half*
    int N, int B,
    double* dE,              // double[B]
    cudaStream_t stream);

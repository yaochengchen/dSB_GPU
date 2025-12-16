// cdsb_fastshare.cu

#include "cdsb_fastshare.h"

#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>

// -----------------------------------------
// Kernel: fused update loop (FP32)
// Layout: x[row*B + batch], y[row*B + batch], J[row*N + col] (row-major)
// One block per batch, shared stage x/y, each thread handles multiple rows.
// (Modified to match the style of the 1st reference code.)
// -----------------------------------------
template<int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS, 2)
cdsb_fused_dense_fastshare_kernel_fp32(
    float* __restrict__ y,          // [N*B]
    float* __restrict__ x,          // [N*B]
    const float* __restrict__ J,    // [N*N] row-major
    const float* __restrict__ p,    // [iters]
    float delta, float xi, float dt,
    int N, int B, int iters
) {
  int batch = (int)blockIdx.x;
  if (batch >= B) return;

  int tid = (int)threadIdx.x;

  // shared: sx[N], sy[N]
  extern __shared__ float smem[];
  float* sx = smem;       // N
  float* sy = smem + N;   // N

  // each thread handles multiple rows
  int rows_per_thread = (N + BLOCK_THREADS - 1) / BLOCK_THREADS;

  // load x/y to shared
  for (int i = 0; i < rows_per_thread; ++i) {
    int row = tid + i * BLOCK_THREADS;
    if (row < N) {
      int g = row * B + batch;
      sx[row] = x[g];
      sy[row] = y[g];
    }
  }
  __syncthreads();

  // iterations
  for (int it = 0; it < iters; ++it) {
    float p_i = p[it];

    // each thread updates its rows
    for (int i = 0; i < rows_per_thread; ++i) {
      int row = tid + i * BLOCK_THREADS;
      if (row < N) {
        float xv = sx[row];
        float yv = sy[row];

        // dense J @ sign(x) using shared sx, sign computed on-the-fly
        const float* Jrow = J + (size_t)row * (size_t)N;
        float acc = 0.f;
        for (int col = 0; col < N; ++col) {
          float vx = sx[col];
          float s  = (vx > 0.f) ? 1.f : ((vx < 0.f) ? -1.f : 0.f);
          acc += Jrow[col] * s;
        }

        // update (Python-aligned)
        // y += (-(delta-p)*x + xi*(J@sign(x))) * dt
        yv += (-(delta - p_i) * xv + xi * acc) * dt;

        // x += dt * y * delta
        xv += dt * yv * delta;

        // if |x|>1 => x=sign(x), y=0
        if (fabsf(xv) > 1.f) {
          xv = (xv > 0.f) ? 1.f : -1.f;
          yv = 0.f;
        }

        // write back to shared
        sx[row] = xv;
        sy[row] = yv;
      }
    }

    __syncthreads(); // barrier per iteration (like reference)
  }

  // store back
  for (int i = 0; i < rows_per_thread; ++i) {
    int row = tid + i * BLOCK_THREADS;
    if (row < N) {
      int g = row * B + batch;
      x[g] = sx[row];
      y[g] = sy[row];
    }
  }
}

// -----------------------------------------
// Energy kernel (FP32):
// E[b] = -0.5 * sum_i ( (J * sign(x[:,b]))_i * sign(x_i,b) )
// -----------------------------------------
template<int BLOCK_THREADS>
__global__ void cdsb_energy_kernel_fp32(
    const float* __restrict__ x,     // [N*B]
    const float* __restrict__ J,     // [N*N]
    int N, int B,
    double* __restrict__ E           // [B]
) {
  int batch = (int)blockIdx.x;
  if (batch >= B) return;

  extern __shared__ float smem[];
  float* sgn = smem; // N

  int tid = (int)threadIdx.x;
  for (int i = tid; i < N; i += BLOCK_THREADS) {
    float v = x[i * B + batch];
    float s = (v > 0.f) ? 1.f : ((v < 0.f) ? -1.f : 0.f);
    sgn[i] = s;
  }
  __syncthreads();

  constexpr int WARP = 32;
  int lane = tid & (WARP - 1);
  int warp_id = tid >> 5;
  int warps_per_block = BLOCK_THREADS / WARP;

  double sum = 0.0;

  for (int row = warp_id; row < N; row += warps_per_block) {
    float acc = 0.f;
    int col = lane;
    int stride = WARP;

    const float* Jrow = J + (size_t)row * (size_t)N;
    for (; col < N; col += stride) {
      acc += Jrow[col] * sgn[col];
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, off);
    }

    if (lane == 0) {
      sum += (double)acc * (double)sgn[row];
    }
  }

  __shared__ double block_sum;
  if (tid == 0) block_sum = 0.0;
  __syncthreads();

  if (lane == 0) atomicAdd(&block_sum, sum);
  __syncthreads();

  if (tid == 0) {
    E[batch] = -0.5 * block_sum;
  }
}

// -----------------------------------------
// C-callable launchers
// -----------------------------------------
void cdsb_fused_run_fp32(
    float* dY,
    float* dX,
    const float* dJ,
    const float* dP,
    float delta, float xi, float dt,
    int N, int B, int iters,
    cudaStream_t stream
) {
  // Match reference style: use max threads if possible.
  // NOTE: if your GPU/compile target doesn't allow 1024 threads for some reason,
  // you can lower this back to 512 without changing the kernel logic.
  constexpr int BLOCK = 1024;

  // Now only need sx + sy = 2*N floats
  size_t smem = (size_t)2 * (size_t)N * sizeof(float);

  cdsb_fused_dense_fastshare_kernel_fp32<BLOCK>
      <<<dim3(B, 1, 1), dim3(BLOCK, 1, 1), smem, stream>>>(
          dY, dX, dJ, dP, delta, xi, dt, N, B, iters);
}

void cdsb_energy_fp32(
    const float* dX,
    const float* dJ,
    int N, int B,
    double* dE,
    cudaStream_t stream
) {
  constexpr int BLOCK = 512;
  size_t smem = (size_t)N * sizeof(float);

  cdsb_energy_kernel_fp32<BLOCK>
      <<<dim3(B, 1, 1), dim3(BLOCK, 1, 1), smem, stream>>>(dX, dJ, N, B, dE);
}

// -----------------------------------------
// CDSB class implementation (FP32)
// -----------------------------------------
CDSB::CDSB(const Mat& J, int batch_size, int n_iter, float delta, float xi, float dt)
    : N_((int)J.rows()),
      B_(batch_size),
      iters_(n_iter),
      delta_(delta),
      xi_(xi),
      dt_(dt) {
  if (J.rows() != J.cols()) throw std::runtime_error("CDSB: J must be square");
  if (N_ <= 0 || B_ <= 0 || iters_ <= 0) throw std::runtime_error("CDSB: invalid N/B/iters");

  x = Mat::Zero(N_, B_);

  pack_J_to_float_(J);

  // Python-aligned auto xi:
  //   xi = 0.5 * sqrt(N-1) / sqrt(sum(J^2))
  // Trigger when xi is NAN or 0.
  auto_set_xi_from_J_();

  build_p_schedule_();
  init_random_xy_();
  alloc_device_();
  upload_all_();
}

CDSB::~CDSB() { free_device_(); }

void CDSB::alloc_device_() {
  CDSB_CUDA_CHECK(cudaMalloc(&dJ_, (size_t)N_ * (size_t)N_ * sizeof(float)));
  CDSB_CUDA_CHECK(cudaMalloc(&dx_, (size_t)N_ * (size_t)B_ * sizeof(float)));
  CDSB_CUDA_CHECK(cudaMalloc(&dy_, (size_t)N_ * (size_t)B_ * sizeof(float)));
  CDSB_CUDA_CHECK(cudaMalloc(&dp_, (size_t)iters_ * sizeof(float)));
}

void CDSB::free_device_() noexcept {
  if (dJ_) cudaFree(dJ_);
  if (dx_) cudaFree(dx_);
  if (dy_) cudaFree(dy_);
  if (dp_) cudaFree(dp_);
  dJ_ = dx_ = dy_ = dp_ = nullptr;
}

void CDSB::build_p_schedule_() {
  hp_.resize((size_t)iters_);
  if (iters_ == 1) {
    hp_[0] = 0.f;
  } else {
    for (int i = 0; i < iters_; ++i) {
      hp_[i] = (float)i / (float)(iters_ - 1);
    }
  }
}

void CDSB::pack_J_to_float_(const Mat& J) {
  hJ_.resize((size_t)N_ * (size_t)N_);
  for (int r = 0; r < N_; ++r) {
    for (int c = 0; c < N_; ++c) {
      hJ_[(size_t)r * (size_t)N_ + (size_t)c] = (float)J(r, c);
    }
  }
}

void CDSB::auto_set_xi_from_J_() {
  // Only auto-set if xi_ is not provided (NAN) or equals 0.
  if (std::isfinite(xi_) && xi_ != 0.0f) return;

  double sumsq = 0.0;
  for (size_t k = 0; k < hJ_.size(); ++k) {
    double v = (double)hJ_[k];
    sumsq += v * v;
  }

  if (sumsq == 0.0) {
    xi_ = 0.0f; // degenerate; you may prefer throw
    return;
  }

  double num = 0.5 * std::sqrt((double)(N_ - 1));
  double den = std::sqrt(sumsq);
  xi_ = (float)(num / den);
}

// Python-aligned init: x,y = 0.02*(rand-0.5) => Uniform(-0.01, 0.01)
void CDSB::init_random_xy_() {
  std::mt19937 gen(12345); // keep fixed seed for reproducibility
  std::uniform_real_distribution<double> dist(-0.01, 0.01);

  hx_.resize((size_t)N_ * (size_t)B_);
  hy_.resize((size_t)N_ * (size_t)B_);

  for (int r = 0; r < N_; ++r) {
    for (int b = 0; b < B_; ++b) {
      double xv = dist(gen);
      double yv = dist(gen);

      x(r, b) = xv;

      size_t idx = (size_t)r * (size_t)B_ + (size_t)b;
      hx_[idx] = (float)xv;
      hy_[idx] = (float)yv;
    }
  }

  gpu_state_dirty_ = false;
}

void CDSB::pack_xy_to_float_() {
  if ((int)x.rows() != N_ || (int)x.cols() != B_) {
    throw std::runtime_error("CDSB: host x shape changed unexpectedly");
  }
  hx_.resize((size_t)N_ * (size_t)B_);
  for (int r = 0; r < N_; ++r) {
    for (int b = 0; b < B_; ++b) {
      size_t idx = (size_t)r * (size_t)B_ + (size_t)b;
      hx_[idx] = (float)x(r, b);
    }
  }
  gpu_state_dirty_ = false;
}

void CDSB::unpack_x_from_float_() {
  hx_.resize((size_t)N_ * (size_t)B_);
  CDSB_CUDA_CHECK(cudaMemcpy(hx_.data(), dx_, hx_.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (int r = 0; r < N_; ++r) {
    for (int b = 0; b < B_; ++b) {
      size_t idx = (size_t)r * (size_t)B_ + (size_t)b;
      x(r, b) = (double)hx_[idx];
    }
  }
}

void CDSB::upload_all_() {
  CDSB_CUDA_CHECK(cudaMemcpy(dJ_, hJ_.data(), hJ_.size()*sizeof(float), cudaMemcpyHostToDevice));
  CDSB_CUDA_CHECK(cudaMemcpy(dx_, hx_.data(), hx_.size()*sizeof(float), cudaMemcpyHostToDevice));
  CDSB_CUDA_CHECK(cudaMemcpy(dy_, hy_.data(), hy_.size()*sizeof(float), cudaMemcpyHostToDevice));
  CDSB_CUDA_CHECK(cudaMemcpy(dp_, hp_.data(), hp_.size()*sizeof(float), cudaMemcpyHostToDevice));
}

void CDSB::upload_x_if_dirty_() const {
  if (!gpu_state_dirty_) return;
  const_cast<CDSB*>(this)->pack_xy_to_float_();
  CDSB_CUDA_CHECK(cudaMemcpy(dx_, hx_.data(), hx_.size()*sizeof(float), cudaMemcpyHostToDevice));
}

void CDSB::update() {
  cudaStream_t stream = 0;

  cdsb_fused_run_fp32(
      dy_, dx_, dJ_, dp_,
      delta_, xi_, dt_,
      N_, B_, iters_,
      stream);

  CDSB_CUDA_CHECK(cudaGetLastError());
  CDSB_CUDA_CHECK(cudaStreamSynchronize(stream));

  unpack_x_from_float_();
}

std::vector<double> CDSB::calc_energy() const {
  upload_x_if_dirty_();

  cudaStream_t stream = 0;
  double* dE = nullptr;
  CDSB_CUDA_CHECK(cudaMalloc(&dE, (size_t)B_ * sizeof(double)));

  cdsb_energy_fp32(dx_, dJ_, N_, B_, dE, stream);

  CDSB_CUDA_CHECK(cudaGetLastError());
  CDSB_CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> hE((size_t)B_);
  CDSB_CUDA_CHECK(cudaMemcpy(hE.data(), dE, (size_t)B_ * sizeof(double), cudaMemcpyDeviceToHost));
  cudaFree(dE);
  return hE;
}

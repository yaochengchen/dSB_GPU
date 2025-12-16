// cdsb_fastshare.cu

#include "cdsb_fastshare.h"

#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>

// -----------------------------------------
// Kernel: fused update loop (FP16 storage, FP32 compute)
// Layout: x[row*B + batch], y[row*B + batch], J[row*N + col] (row-major)
// -----------------------------------------
template<int BLOCK_THREADS>
__global__ void cdsb_fused_dense_fastshare_kernel_fp16(
    __half* __restrict__ y,          // [N*B] FP16
    __half* __restrict__ x,          // [N*B] FP16
    const __half* __restrict__ J,    // [N*N] FP16 row-major
    const __half* __restrict__ p,    // [iters] FP16
    float delta, float xi, float dt,
    int N, int B, int iters
) {
  int batch = (int)blockIdx.x;
  if (batch >= B) return;

  // Shared memory in FP32 for stable compute
  extern __shared__ float smem[];
  float* x0  = smem;          // N
  float* y0  = smem + N;      // N
  float* x1  = smem + 2*N;    // N
  float* y1  = smem + 3*N;    // N
  float* sgn = smem + 4*N;    // N  (-1,0,1) as float

  int tid = (int)threadIdx.x;

  // load x/y (half->float)
  for (int row = tid; row < N; row += BLOCK_THREADS) {
    int g = row * B + batch;
    x0[row] = __half2float(x[g]);
    y0[row] = __half2float(y[g]);
  }
  __syncthreads();

  constexpr int WARP = 32;
  int lane = tid & (WARP - 1);
  int warp_id = tid >> 5;
  int warps_per_block = BLOCK_THREADS / WARP;

  for (int it = 0; it < iters; ++it) {
    float p_i = __half2float(p[it]);

    // sign(x0) into shared
    for (int col = tid; col < N; col += BLOCK_THREADS) {
      float v = x0[col];
      sgn[col] = (v > 0.f) ? 1.f : ((v < 0.f) ? -1.f : 0.f);
    }
    __syncthreads();

    // each warp handles rows
    for (int row = warp_id; row < N; row += warps_per_block) {
      float acc = 0.f;

      const __half* Jrow = J + (size_t)row * (size_t)N;

      // lane-strided dot
      for (int col = lane; col < N; col += WARP) {
        float jv = __half2float(Jrow[col]);
        acc += jv * sgn[col];
      }

      // warp reduce
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
      }

      if (lane == 0) {
        float xv = x0[row];
        float yv = y0[row];

        // Python: y += (-(delta-p)*x + xi*(J@sign(x))) * dt
        yv += (-(delta - p_i) * xv + xi * acc) * dt;

        // Python: x += dt * y * delta
        xv += dt * yv * delta;

        // Python: if |x|>1 => x=sign(x), y=0
        if (fabsf(xv) > 1.f) {
          xv = (xv > 0.f) ? 1.f : -1.f;
          yv = 0.f;
        }

        x1[row] = xv;
        y1[row] = yv;
      }
    }

    __syncthreads();

    // swap
    for (int row = tid; row < N; row += BLOCK_THREADS) {
      x0[row] = x1[row];
      y0[row] = y1[row];
    }
    __syncthreads();
  }

  // store back (float->half)
  for (int row = tid; row < N; row += BLOCK_THREADS) {
    int g = row * B + batch;
    x[g] = __float2half_rn(x0[row]);
    y[g] = __float2half_rn(y0[row]);
  }
}

// -----------------------------------------
// Energy kernel (FP16 storage):
// E[b] = -0.5 * sum_i ( (J * sign(x[:,b]))_i * sign(x_i,b) )
// sign computed in float; dot uses float accum; final sum in double.
// -----------------------------------------
template<int BLOCK_THREADS>
__global__ void cdsb_energy_kernel_fp16(
    const __half* __restrict__ x,     // [N*B]
    const __half* __restrict__ J,     // [N*N]
    int N, int B,
    double* __restrict__ E           // [B]
) {
  int batch = (int)blockIdx.x;
  if (batch >= B) return;

  extern __shared__ float smem[];
  float* sgn = smem; // N

  int tid = (int)threadIdx.x;
  for (int i = tid; i < N; i += BLOCK_THREADS) {
    float v = __half2float(x[i * B + batch]);
    sgn[i] = (v > 0.f) ? 1.f : ((v < 0.f) ? -1.f : 0.f);
  }
  __syncthreads();

  constexpr int WARP = 32;
  int lane = tid & (WARP - 1);
  int warp_id = tid >> 5;
  int warps_per_block = BLOCK_THREADS / WARP;

  double sum = 0.0;

  for (int row = warp_id; row < N; row += warps_per_block) {
    float acc = 0.f;

    const __half* Jrow = J + (size_t)row * (size_t)N;
    for (int col = lane; col < N; col += WARP) {
      acc += __half2float(Jrow[col]) * sgn[col];
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
void cdsb_fused_run_fp16(
    __half* dY,
    __half* dX,
    const __half* dJ,
    const __half* dP,
    float delta, float xi, float dt,
    int N, int B, int iters,
    cudaStream_t stream
) {
  constexpr int BLOCK = 1024;
  size_t smem = (size_t)5 * (size_t)N * sizeof(float); // shared float

  cdsb_fused_dense_fastshare_kernel_fp16<BLOCK>
      <<<dim3(B, 1, 1), dim3(BLOCK, 1, 1), smem, stream>>>(
          dY, dX, dJ, dP, delta, xi, dt, N, B, iters);
}

void cdsb_energy_fp16(
    const __half* dX,
    const __half* dJ,
    int N, int B,
    double* dE,
    cudaStream_t stream
) {
  constexpr int BLOCK = 1024;
  size_t smem = (size_t)N * sizeof(float);

  cdsb_energy_kernel_fp16<BLOCK>
      <<<dim3(B, 1, 1), dim3(BLOCK, 1, 1), smem, stream>>>(dX, dJ, N, B, dE);
}

// -----------------------------------------
// CDSB class implementation (J host double -> GPU half)
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

  pack_J_to_half_(J);

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
  CDSB_CUDA_CHECK(cudaMalloc(&dJ_, (size_t)N_ * (size_t)N_ * sizeof(__half)));
  CDSB_CUDA_CHECK(cudaMalloc(&dx_, (size_t)N_ * (size_t)B_ * sizeof(__half)));
  CDSB_CUDA_CHECK(cudaMalloc(&dy_, (size_t)N_ * (size_t)B_ * sizeof(__half)));
  CDSB_CUDA_CHECK(cudaMalloc(&dp_, (size_t)iters_ * sizeof(__half)));
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
    hp_[0] = __float2half_rn(0.f);
  } else {
    for (int i = 0; i < iters_; ++i) {
      float pv = (float)i / (float)(iters_ - 1);
      hp_[i] = __float2half_rn(pv);
    }
  }
}

void CDSB::pack_J_to_half_(const Mat& J) {
  hJ_.resize((size_t)N_ * (size_t)N_);
  for (int r = 0; r < N_; ++r) {
    for (int c = 0; c < N_; ++c) {
      float v = (float)J(r, c);
      hJ_[(size_t)r * (size_t)N_ + (size_t)c] = __float2half_rn(v);
    }
  }
}

void CDSB::auto_set_xi_from_J_() {
  // Only auto-set if xi_ is not provided (NAN) or equals 0.
  if (std::isfinite(xi_) && xi_ != 0.0f) return;

  double sumsq = 0.0;
  for (size_t k = 0; k < hJ_.size(); ++k) {
    double v = (double)__half2float(hJ_[k]);
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
      hx_[idx] = __float2half_rn((float)xv);
      hy_[idx] = __float2half_rn((float)yv);
    }
  }

  gpu_state_dirty_ = false;
}

void CDSB::pack_x_to_half_() {
  if ((int)x.rows() != N_ || (int)x.cols() != B_) {
    throw std::runtime_error("CDSB: host x shape changed unexpectedly");
  }
  hx_.resize((size_t)N_ * (size_t)B_);
  for (int r = 0; r < N_; ++r) {
    for (int b = 0; b < B_; ++b) {
      size_t idx = (size_t)r * (size_t)B_ + (size_t)b;
      hx_[idx] = __float2half_rn((float)x(r, b));
    }
  }
  gpu_state_dirty_ = false;
}

void CDSB::unpack_x_from_half_() {
  hx_.resize((size_t)N_ * (size_t)B_);
  CDSB_CUDA_CHECK(cudaMemcpy(hx_.data(), dx_, hx_.size() * sizeof(__half), cudaMemcpyDeviceToHost));
  for (int r = 0; r < N_; ++r) {
    for (int b = 0; b < B_; ++b) {
      size_t idx = (size_t)r * (size_t)B_ + (size_t)b;
      x(r, b) = (double)__half2float(hx_[idx]);
    }
  }
}

void CDSB::upload_all_() {
  CDSB_CUDA_CHECK(cudaMemcpy(dJ_, hJ_.data(), hJ_.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CDSB_CUDA_CHECK(cudaMemcpy(dx_, hx_.data(), hx_.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CDSB_CUDA_CHECK(cudaMemcpy(dy_, hy_.data(), hy_.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CDSB_CUDA_CHECK(cudaMemcpy(dp_, hp_.data(), hp_.size() * sizeof(__half), cudaMemcpyHostToDevice));
}

void CDSB::upload_x_if_dirty_() const {
  if (!gpu_state_dirty_) return;
  const_cast<CDSB*>(this)->pack_x_to_half_();
  CDSB_CUDA_CHECK(cudaMemcpy(dx_, hx_.data(), hx_.size() * sizeof(__half), cudaMemcpyHostToDevice));
}

void CDSB::update() {
  cudaStream_t stream = 0;

  cdsb_fused_run_fp16(
      dy_, dx_, dJ_, dp_,
      delta_, xi_, dt_,
      N_, B_, iters_,
      stream);

  CDSB_CUDA_CHECK(cudaGetLastError());
  CDSB_CUDA_CHECK(cudaStreamSynchronize(stream));

  unpack_x_from_half_();
}

std::vector<double> CDSB::calc_energy() const {
  upload_x_if_dirty_();

  cudaStream_t stream = 0;
  double* dE = nullptr;
  CDSB_CUDA_CHECK(cudaMalloc(&dE, (size_t)B_ * sizeof(double)));

  cdsb_energy_fp16(dx_, dJ_, N_, B_, dE, stream);

  CDSB_CUDA_CHECK(cudaGetLastError());
  CDSB_CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> hE((size_t)B_);
  CDSB_CUDA_CHECK(cudaMemcpy(hE.data(), dE, (size_t)B_ * sizeof(double), cudaMemcpyDeviceToHost));
  cudaFree(dE);
  return hE;
}

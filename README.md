# CDSB FastShare (CUDA, no Torch)

A **pure C++ / CUDA** implementation of a CDSB-style dynamics solver with dense coupling matrix **J**, designed for NVIDIA GPUs (e.g., H100) without PyTorch or pybind dependencies.

---

## Typical Problem Scales

This implementation is designed for the following typical ranges:

| Problem Size | N (Variables) | B (Batch Size) | Use Case |
|--------------|---------------|----------------|-----------|
| Small | N ≈ 1024 | B ≈ 64 | High throughput, many parallel solves |
| Medium | N ≈ 4096 | B ≈ 8 | Balanced memory/compute |
| Large | N ≈ 8192 | B ≈ 1 | Single large problem instance |

> **Note:** Shared memory usage scales as `5*N*sizeof(float)` per block. Large N may exceed GPU limits (typically ~48KB per block on consumer GPUs, ~227KB on H100).

---

## Quick Start

### 1. Download Test Data

Download QPLIB benchmark instances from [http://qplib.zib.de/](http://qplib.zib.de/)

```bash
# Example: download a QPLIB instance
wget http://qplib.zib.de/instances/[instance_name].qplib
```

Place `.qplib` files in the project root directory.

### 2. Build

```bash
make
```

Requirements:
- NVCC (CUDA Toolkit 11.0+)
- Eigen3 headers
- g++ with C++17 support

### 3. Run

```bash
./app
```

The program will automatically read QPLIB instances from the current directory and run the solver.

---

## Key Features

- **Host API uses Eigen** (`Eigen::MatrixXd`) for convenience
- **GPU storage uses FP16 (`__half`)** for `J`, `x`, `y`, and schedule `p` to reduce memory bandwidth
- **Compute uses FP32** in shared memory and accumulators for numerical stability
- **One CUDA block per batch element** (grid.x = B) with **warp-per-row** parallelization
- **No PyTorch/Python dependencies** - pure C++/CUDA

---

## Repository Structure

```
.
├── README.md                     # This file
├── Makefile                      # Build configuration
├── cdsb_fastshare.h             # CDSB CUDA interface header
├── cdsb_fastshare.cu            # CDSB CUDA kernels + implementation
├── cdsb_fasthare_qplib.cpp      # QPLIB wrapper/entry point
├── fasthare_api.cpp             # FastHare API implementation
├── src/
│   ├── fasthare.cpp             # FastHare algorithm
│   └── graph.cpp                # Graph utilities
└── *.qplib                       # QPLIB problem instances (download separately)
```

---

## Algorithm Overview

### State Variables (per batch)

- `x ∈ R^(N×B)` — positions
- `y ∈ R^(N×B)` — velocities  
- `J ∈ R^(N×N)` — dense coupling matrix (shared across batches)
- `p[it]` — schedule over iterations

All GPU tensors use **row-major packed layout**:
- `x[row * B + batch]`
- `y[row * B + batch]`
- `J[row * N + col]`

### Update Equations

Let `s = sign(x)` elementwise, with `sign(0) = 0`.

For each iteration `it`:

1. **Compute coupling term:**  
   `acc[row] = sum_col J[row, col] * s[col]`

2. **Update dynamics:**
   ```
   y[row] += (-(delta - p[it]) * x[row] + xi * acc[row]) * dt
   x[row] += dt * y[row] * delta
   ```

3. **Clamp rule:**  
   If `|x[row]| > 1`: set `x[row] = sign(x[row])` and `y[row] = 0`

---

## GPU Parallelization

### Grid / Blocks

- **gridDim.x = B** — one CUDA block per batch element
- **blockIdx.x** identifies which batch element to process

### Threads / Warps

- `BLOCK_THREADS = 512` threads per block (default)
- **Warp-per-row** strategy: each warp processes rows `warp_id, warp_id + warps_per_block, ...`

### Shared Memory Strategy

Each block stages FP32 arrays for a single batch:
- `x0[N], y0[N]` — current state
- `x1[N], y1[N]` — next state (ping-pong buffers)
- `sgn[N]` — sign(x)

**Total shared memory per block:** `5 * N * sizeof(float)`

> ⚠️ **Large N Warning:** Shared memory grows linearly with N and can exceed GPU limits.

---

## Precision Policy

| Data | Storage Format | Compute Format |
|------|----------------|----------------|
| Host input `J` | `double` (Eigen) | — |
| Device `J`, `x`, `y`, `p` | `__half` (FP16) | `float` (FP32) |
| Kernel computation | — | `float` (FP32) |

**Rationale:** FP16 storage reduces memory bandwidth while FP32 compute maintains numerical stability.

---

## Public API

### Construction

```cpp
#include "cdsb_fastshare.h"

// J: square N×N matrix
Eigen::MatrixXd J = ...; 

CDSB solver(J, batch_size, n_iter, delta, xi, dt);
```

**Parameters:**
- `J` — `Eigen::MatrixXd` (N×N coupling matrix)
- `batch_size` — number of parallel solves (B)
- `n_iter` — iteration count
- `delta`, `dt` — dynamics constants
- `xi` — coupling gain (if `NaN` or `0`, auto-computed as `0.5 * sqrt(N-1) / sqrt(sum(J^2))`)

### Run Solver

```cpp
solver.update();  // Execute n_iter steps on GPU
// After update(), solver.x contains results (N×B Eigen matrix)
```

### Compute Energy

```cpp
std::vector<double> E = solver.calc_energy();
// Returns energy for each batch: E[b] = -0.5 * sum_i((J*sign(x[:,b]))_i * sign(x[i,b]))
```

---

## Initialization

- **RNG seed:** Fixed at `12345` for reproducibility
- **Initial state:** `x, y ~ Uniform(-0.01, 0.01)`
- **Schedule:** Linear interpolation
  - If `iters == 1`: `p[0] = 0`
  - Otherwise: `p[it] = it / (iters - 1)`

---

## Build Configuration

### Makefile Overview

The project uses a Makefile with the following configuration:

**Compilers:**
- `g++` for C++ sources
- `nvcc` for CUDA sources

**Compilation Flags:**
- C++17 standard
- `-O2` optimization
- `--use_fast_math` for CUDA (fast math optimizations)
- `-gencode arch=compute_90,code=sm_90` (for H100)

**Source Files:**
- C++ sources: `cdsb_fasthare_qplib.cpp`, `fasthare_api.cpp`, `src/fasthare.cpp`, `src/graph.cpp`
- CUDA sources: `cdsb_fastshare.cu`

**Dependencies:**
- Eigen3 (typically in `/usr/include/eigen3`)
- CUDA Toolkit (typically in `/usr/local/cuda`)

### Architecture-Specific Builds

Adjust the architecture flag in Makefile for your GPU:
- **H100:** `-gencode arch=compute_90,code=sm_90`
- **A100:** `-gencode arch=compute_80,code=sm_80`
- **RTX 30xx:** `-gencode arch=compute_86,code=sm_86`

Or use the simpler form: `-arch=sm_90`

---

## Performance Considerations

### Limitations

1. **Shared memory scaling:**
   - `5*N*sizeof(float)` per block
   - H100: ~227KB shared memory/block → max N ≈ 11,600
   - Consumer GPUs: ~48KB → max N ≈ 2,400

2. **Occupancy:**
   - Large shared memory usage reduces active blocks per SM

3. **Batch parallelism:**
   - GPU utilization requires `B` to be large enough (typically B ≥ 16 for good occupancy)

4. **Memory bandwidth:**
   - FP16 storage helps, but dense J access can bottleneck for very large N

### Optimization Ideas

- **Reduce shared memory:** Convert `sgn` to `int8`
- **Tile N dimension:** Allow multiple blocks per batch for very large N
- **Vectorization:** Use `half2` for paired loads (requires handling odd N)
- **Cooperative groups:** Optimize large dot products with split-K reduction

---

## Example Usage

```cpp
#include "cdsb_fastshare.h"
#include <iostream>

int main() {
    // Create random coupling matrix
    int N = 1024;
    int B = 32;
    Eigen::MatrixXd J = Eigen::MatrixXd::Random(N, N);
    
    // Initialize solver
    CDSB solver(J, B, 1000, 1.0, NAN, 0.01);
    
    // Run dynamics
    solver.update();
    
    // Get results
    Eigen::MatrixXd x = solver.x;  // N×B matrix
    std::vector<double> energies = solver.calc_energy();
    
    std::cout << "Final energies: ";
    for (double E : energies) {
        std::cout << E << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

---

## References

- **QPLIB:** Quadratic Programming Library — [http://qplib.zib.de/](http://qplib.zib.de/)
- CDSB dynamics: Continuous-time dynamical system for quadratic optimization

---

## License

This is a research implementation. Verify correctness and numerical stability for your specific use case and parameter ranges.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cdsb_fastshare,
  title = {CDSB FastShare: CUDA Implementation of CDSB Dynamics},
  author = {[Your Name]},
  year = {2024},
  url = {https://github.com/[your-username]/cdsb-fastshare}
}
```

---

## Contributing

Contributions welcome! Please open issues or pull requests for:
- Bug fixes
- Performance improvements
- Extended N support via tiling
- Additional optimization algorithms

---

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com].
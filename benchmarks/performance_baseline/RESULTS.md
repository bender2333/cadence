# Multi-Scale Performance Baseline Comparison: CPU vs Ascend NPU

Comprehensive multi-scale performance baseline comparison across CPU (`numpy` float64 / `torch` CPU float32) and Huawei Ascend NPU (`torch_npu` float32) on the Orange Pi AIpro single-board computer (Ascend 310B4).

All 48 configurations across network scales $N \in \{64, 256, 1024, 2048\}$ and batch regimes $B \in \{1, 4, 16, 64\}$ executed on actual hardware. All runs maintained equation residual $\le 0.003$ with 100% numerical qualification and zero free-answer refusals. The independent verifier confirmed all 146 checks without failure.

## 1. Sources and Environment

- Commit: `origin/main` baseline on branch `v1/11-perf-baseline`
- Hardware: Orange Pi AIpro SBC
  - NPU: Huawei Ascend 310B4 AI Core (CANN 8.0.RC3, driver 25.2.0)
  - CPU: Quad-core ARM Cortex-A55 @ 1.6 GHz (2 cores exposed to test cgroup)
  - RAM: 7.4 GiB LPDDR4X (host RSS cgroup capped at 5.2 GiB)
  - OS: Linux 5.10.0+ aarch64
- Software stack:
  - Python: 3.11.15
  - PyTorch: 2.4.0
  - torch_npu: 2.4.0
  - NumPy: 2.4.6
  - CANN: 8.0.RC3
- Verifiable receipt: [`results/baseline-comparison-receipt.json.gz`](results/baseline-comparison-receipt.json.gz)

---

## 2. Core Latency and Throughput Comparisons

### 2.1 Free Relaxation (`NeuralGraph.equilibrate`)

Warm steady-state latencies (milliseconds, p50 / p95) and throughput (relaxation steps per second) to residual $\le 0.003$:

| Scale ($N$) | Batch ($B$) | CPU (NumPy f64) p50 / p95 (ms) | Torch CPU (f32) p50 / p95 (ms) | Ascend NPU (f32) p50 / p95 (ms) | Speedup (NPU / CPU) | CPU Steps/s | NPU Steps/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 1.49 / 1.65 | 33.15 / 33.39 | 84.97 / 93.33 | 0.02x | 21,024 | 373 |
| 64 | 4 | 1.90 / 1.99 | 34.73 / 34.87 | 122.45 / 125.66 | 0.02x | 16,721 | 265 |
| 64 | 16 | 3.65 / 3.76 | 36.88 / 37.12 | 121.26 / 126.22 | 0.03x | 8,715 | 268 |
| 64 | 64 | 10.46 / 10.55 | 44.59 / 44.81 | 125.54 / 133.12 | 0.08x | 3,055 | 259 |
| 256 | 1 | 2.05 / 2.07 | 33.88 / 33.98 | 88.72 / 119.64 | 0.02x | 15,646 | 340 |
| 256 | 4 | 3.60 / 3.65 | 36.69 / 36.95 | 111.83 / 121.44 | 0.03x | 8,877 | 279 |
| 256 | 16 | 10.51 / 10.61 | 43.73 / 43.97 | 114.00 / 121.59 | 0.09x | 3,038 | 278 |
| 256 | 64 | 39.21 / 39.55 | 70.17 / 70.77 | 122.73 / 128.14 | 0.32x | 814 | 263 |
| 1024 | 1 | 4.54 / 4.59 | 38.51 / 38.76 | 87.70 / 93.49 | 0.05x | 7,058 | 373 |
| 1024 | 4 | 11.99 / 12.17 | 45.94 / 46.11 | 119.38 / 126.94 | 0.10x | 2,659 | 264 |
| 1024 | 16 | 42.75 / 43.93 | 72.40 / 75.16 | 126.53 / 128.77 | 0.34x | 745 | 254 |
| **1024** | **64** | **168.21 / 168.73** | **137.39 / 140.92** | **138.68 / 146.82** | **1.21x** | **191** | **232** |
| 2048 | 1 | 9.41 / 9.53 | 47.35 / 47.54 | 91.53 / 97.11 | 0.10x | 3,394 | 365 |
| 2048 | 4 | 27.75 / 27.95 | 63.26 / 63.42 | 120.68 / 133.44 | 0.23x | 1,152 | 264 |
| 2048 | 16 | 90.32 / 91.18 | 116.30 / 119.81 | 134.07 / 142.57 | 0.67x | 354 | 241 |
| **2048** | **64** | **349.84 / 354.39** | **244.97 / 257.66** | **169.47 / 179.06** | **2.06x** | **91** | **189** |

---

### 2.2 Synaptic Plasticity (`Learner.step`)

End-to-end learning step latency (free phase, nudged phases, contrast, parameter step):

| Scale ($N$) | Batch ($B$) | CPU p50 / p95 (ms) | Torch CPU p50 / p95 (ms) | Ascend NPU p50 / p95 (ms) | Speedup (NPU / CPU) | CPU Param Updates/s | NPU Param Updates/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 4.50 / 4.64 | 46.73 / 47.12 | 117.43 / 122.87 | 0.04x | 137,120 | 5,266 |
| 64 | 4 | 5.34 / 5.61 | 47.13 / 47.19 | 155.81 / 158.49 | 0.03x | 114,323 | 3,983 |
| 64 | 16 | 8.68 / 8.77 | 51.86 / 53.12 | 163.00 / 171.05 | 0.05x | 71,738 | 3,763 |
| 64 | 64 | 23.20 / 23.48 | 64.76 / 65.23 | 177.07 / 180.50 | 0.13x | 26,618 | 3,500 |
| 256 | 1 | 5.53 / 5.55 | 42.36 / 43.43 | 113.91 / 118.48 | 0.05x | 509,215 | 25,083 |
| 256 | 4 | 8.09 / 8.30 | 45.36 / 47.09 | 143.23 / 157.82 | 0.06x | 344,123 | 19,281 |
| 256 | 16 | 19.28 / 19.58 | 53.90 / 54.15 | 145.73 / 150.99 | 0.13x | 144,999 | 19,092 |
| 256 | 64 | 65.12 / 65.22 | 81.76 / 82.01 | 156.74 / 159.37 | 0.42x | 43,214 | 17,937 |
| 1024 | 1 | 14.42 / 14.62 | 49.73 / 50.63 | 124.94 / 126.52 | 0.12x | 2,897,883 | 337,357 |
| 1024 | 4 | 24.02 / 24.37 | 57.22 / 58.48 | 151.36 / 164.78 | 0.16x | 1,741,345 | 269,386 |
| 1024 | 16 | 65.24 / 65.91 | 81.35 / 82.07 | 162.65 / 165.98 | 0.40x | 639,668 | 262,004 |
| **1024** | **64** | **219.72 / 224.21** | **141.81 / 142.46** | **184.32 / 185.53** | **1.19x** | **189,924** | **228,563** |
| 2048 | 1 | 38.56 / 39.94 | 79.14 / 79.65 | 261.12 / 267.49 | 0.15x | 4,285,450 | 646,355 |
| 2048 | 4 | 63.38 / 65.19 | 93.94 / 94.87 | 293.80 / 295.93 | 0.22x | 2,603,747 | 561,996 |
| 2048 | 16 | 142.06 / 143.23 | 144.99 / 201.69 | 308.07 / 314.73 | 0.46x | 1,169,945 | 534,201 |
| **2048** | **64** | **473.07 / 476.99** | **267.96 / 270.53** | **363.63 / 374.79** | **1.30x** | **349,287** | **450,546** |

---

### 2.3 Teaching Contrast Computation (`Learner.contrast`)

Throughput of two-phase contrast differences (throughput computed from 5-iteration mean latency):

| Scale ($N$) | Batch ($B$) | Synapses | CPU p50 (ms) | Torch CPU p50 (ms) | Ascend NPU p50 (ms) | CPU Synapses/s | NPU Synapses/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 555 | 0.34 | 1.35 | 3.35 | 1,545,800 | 164,080 |
| 64 | 4 | 555 | 0.37 | 1.38 | 4.27 | 1,479,343 | 129,529 |
| 64 | 16 | 555 | 0.40 | 1.42 | 4.31 | 1,361,990 | 127,252 |
| 64 | 64 | 555 | 0.57 | 1.51 | 4.24 | 967,092 | 50,257 |
| 256 | 1 | 2,552 | 0.40 | 1.43 | 3.85 | 6,280,421 | 611,652 |
| 256 | 4 | 2,552 | 0.44 | 1.47 | 4.65 | 5,787,872 | 550,514 |
| 256 | 16 | 2,552 | 0.62 | 1.59 | 4.81 | 4,083,318 | 513,977 |
| 256 | 64 | 2,552 | 1.43 | 1.97 | 4.52 | 1,776,355 | 558,723 |
| 1024 | 1 | 40,813 | 2.48 | 2.71 | 18.51 | 16,432,510 | 2,206,763 |
| 1024 | 4 | 40,813 | 1.97 | 2.70 | 18.97 | 20,555,468 | 2,142,894 |
| 1024 | 16 | 40,813 | 2.90 | 3.17 | 19.11 | 14,073,896 | 2,135,363 |
| 1024 | 64 | 40,813 | 5.73 | 4.17 | 19.19 | 7,013,739 | 2,113,832 |
| 2048 | 1 | 163,282 | 8.46 | 6.62 | 61.71 | 19,304,523 | 2,644,879 |
| 2048 | 4 | 163,282 | 6.26 | 6.60 | 63.27 | 26,174,224 | 2,580,889 |
| 2048 | 16 | 163,282 | 8.02 | 7.77 | 63.34 | 20,225,033 | 2,575,298 |
| 2048 | 64 | 163,282 | 16.26 | 10.66 | 63.99 | 10,052,195 | 2,551,253 |

*Note: Synapses/s throughput is computed from mean latency across 5 iterations (`synapses / mean_latency_s`), not the p50 latency column.*

---

## 3. Crossover Analysis

The hardware comparison identifies clear operational boundaries between CPU and Ascend 310B4 NPU:

### 3.1 Exact Crossover Thresholds
The crossover points where NPU outpaces CPU occur at **$N \ge 1024$ under high-throughput batching ($B = 64$)**:
1. **$N = 1024, B = 64$**:
   - Free relaxation: NPU takes **138.68 ms** vs CPU **168.21 ms** (**1.21x speedup**; 232 vs 191 steps/s).
   - Synaptic plasticity: NPU takes **184.32 ms** vs CPU **219.72 ms** (**1.19x speedup**; 228,563 vs 189,924 parameter updates/s).
2. **$N = 2048, B = 64$**:
   - Free relaxation: NPU takes **169.47 ms** vs CPU **349.84 ms** (**2.06x speedup**; 189 vs 91 steps/s).
   - Synaptic plasticity: NPU takes **363.63 ms** vs CPU **473.07 ms** (**1.30x speedup**; 450,546 vs 349,287 parameter updates/s).

### 3.2 Low-Latency Edge Control ($B = 1$)
For real-time closed-loop control (robotics / edge priority), **CPU (NumPy/Numba float64) remains the decisively faster execution backend across all tested dimensions**:
- At $N = 64$: CPU settles in **1.49 ms** vs NPU **84.97 ms** (0.02x).
- At $N = 256$: CPU settles in **2.05 ms** vs NPU **88.72 ms** (0.02x).
- At $N = 1024$: CPU settles in **4.54 ms** vs NPU **87.70 ms** (0.05x).
- At $N = 2048$: CPU settles in **9.41 ms** vs NPU **91.53 ms** (0.10x).

**Engineering Recommendation**:
- **Robotics sensory-motor loops ($B = 1$)**: Deploy on CPU (NumPy/Numba) for guaranteed sub-10ms response times.
- **Offline replay, high-dimensional memory, and batched learning ($N \ge 1024, B \ge 64$)**: Offload to Ascend 310B4 NPU tensor cores to achieve up to 2.06x throughput acceleration while offloading CPU cycles.

---

## 4. Hardware Efficiency & Latency Variance on Ascend 310B4

### 4.1 Cold JIT Compilation vs Warm Execution
Ascend CANN performs just-in-time compilation of CCE operator kernels for each unique tensor dimension and batch shape.

| Scale ($N$) | Batch ($B$) | Free Cold (ms) | Free Warm p50 (ms) | Free Warm p95 (ms) | Step Cold (ms) | Step Warm p50 (ms) | Step Warm p95 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 9,644.5 | 84.97 | 93.33 | 5,721.2 | 117.43 | 122.87 |
| 64 | 4 | 8,392.5 | 122.45 | 125.66 | 483.4 | 155.81 | 158.49 |
| 64 | 16 | 8,891.1 | 121.26 | 126.22 | 483.0 | 163.00 | 171.05 |
| 64 | 64 | 8,805.3 | 125.54 | 133.12 | 754.5 | 177.07 | 180.50 |
| 256 | 1 | 78,471.6 | 88.72 | 119.64 | 14,863.0 | 113.91 | 118.48 |
| 256 | 4 | 81,105.2 | 111.83 | 121.44 | 3,391.2 | 143.23 | 157.82 |
| 256 | 16 | 8,860.8 | 114.00 | 121.59 | 471.3 | 145.73 | 150.99 |
| 256 | 64 | 84,912.8 | 122.73 | 128.14 | 3,674.1 | 156.74 | 159.37 |
| 1024 | 1 | 63,346.1 | 87.70 | 93.49 | 52,363.0 | 124.94 | 126.52 |
| 1024 | 4 | 87,499.4 | 119.38 | 126.94 | 3,482.2 | 151.36 | 164.78 |
| 1024 | 16 | 85,377.0 | 126.53 | 128.77 | 3,593.4 | 162.65 | 165.98 |
| 1024 | 64 | 92,622.0 | 138.68 | 146.82 | 3,752.5 | 184.32 | 185.53 |
| 2048 | 1 | 66,003.9 | 91.53 | 97.11 | 14,002.2 | 261.12 | 267.49 |
| 2048 | 4 | 90,175.8 | 120.68 | 133.44 | 3,544.6 | 293.80 | 295.93 |
| 2048 | 16 | 93,394.8 | 134.07 | 142.57 | 3,903.9 | 308.07 | 314.73 |
| 2048 | 64 | 9,493.1 | 169.47 | 179.06 | 751.4 | 363.63 | 374.79 |

- **Cold compilation tax**: First encounter of a new dimension requires 8s to 93s of CCE code generation and compilation. On CPU, cold execution is comparable to or faster than warm p50 (no JIT compilation; first iteration benefits from unpolluted cache and branch predictors).
- **Warm execution stability**: Once compiled into `kernel_meta`, warm steady-state latency exhibits low jitter across iterations: p95/p50 ratio remains $\le 1.10$ across most configurations, with maximum ratio $\le 1.35$ (three exceptions exceeding 1.10: $N=256, B=1$ free relaxation at 1.35; $N=2048, B=4$ free relaxation at 1.11; $N=256, B=4$ synaptic plasticity at 1.10).

---

### 4.2 Memory Utilization

| Scale ($N$) | Batch ($B$) | CPU Host Peak RSS (MiB) | NPU Host Peak RSS (MiB) | NPU Allocated (MiB) | NPU Reserved (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 139.5 | 1,418.5 | 0.029 | 2.0 |
| 64 | 64 | 143.0 | 1,418.5 | 0.311 | 2.0 |
| 256 | 1 | 143.0 | 1,418.5 | 0.113 | 2.0 |
| 256 | 64 | 146.2 | 1,418.5 | 1.220 | 4.0 |
| 1024 | 1 | 151.1 | 1,418.5 | 1.427 | 4.0 |
| 1024 | 64 | 164.9 | 1,418.5 | 5.857 | 10.0 |
| 2048 | 1 | 177.6 | 1,418.5 | 5.530 | 30.0 |
| 2048 | 64 | 202.2 | 1,418.5 | 14.389 | 42.0 |

- **On-device tensor allocations**: Tensor memory consumption on Ascend 310B4 is compact, consuming at most **14.39 MiB** allocated (42.0 MiB reserved by PyTorch caching allocator) at $N = 2048, B = 64$.
- **Host RSS footprint**: CPU execution consumes 139–202 MiB RSS. The NPU arm exhibits a 1,418.5 MiB host RSS baseline attributed to the CANN runtime dynamic libraries, driver memory mappings, and CCE compilation engine, comfortably fitting within the board's 5.2 GiB cgroup limit.

---

## 5. Numerical Qualification

All 48 configurations across CPU, Torch CPU, and Ascend NPU maintained equation residuals strictly below the contract tolerance $0.003$:
- CPU float64: residuals between $6.97 \times 10^{-12}$ and $2.92 \times 10^{-9}$.
- Torch CPU float32: residuals between $4.68 \times 10^{-8}$ and $1.37 \times 10^{-7}$.
- Ascend NPU float32: residuals between $5.47 \times 10^{-8}$ and $1.54 \times 10^{-7}$.

Zero numerical divergence, zero NaN/Inf excursions, and zero solve refusals were witnessed throughout the multi-scale campaign.

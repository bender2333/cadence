# 1. Huawei Ascend NPU Backend via torch_npu

Date: 2026-10-16

## Status

Accepted

## Context

Cadence targets biologically plausible equilibrium world models running on edge robotics and computing hardware. The current codebase supports CPU (NumPy/Numba), CUDA (`backend="torch", device="cuda"`), and Apple Silicon (`backend="mlx"`).
The edge AI deployment host is an Orange Pi AIpro featuring a Huawei Ascend 310B4 NPU with CANN 8.0.RC3.
To enable low-power, high-throughput on-device equilibrium settlement and local learning, Cadence needs native execution on Ascend NPU hardware.

## Decision

1. Extend `backend="torch"` with `device="npu"` (e.g. `npu:0`), delegating tensor operations to Huawei's `torch_npu` plugin.
2. Restrict precision on NPU to `float32` (similar to MPS), rejecting `float64` to avoid emulated or unsupported floating-point paths on the Ascend 310B4 AI Core.
3. Automatically expose NPU availability in `cadence.available_backends()` when `torch_npu` and Ascend devices are detected.
4. Establish `benchmarks/system1_npu/` replicating the `benchmarks/system1_cuda/` evaluation harness, quantifying behavioral preservation, equation residuals, latency percentiles, and throughput.

## Consequences

- Cadence gains acceleration on Ascend hardware without altering top-level API contracts (`Brain.compose`, `NeuralGraph`, `Learner`).
- Running on NPU requires `torch>=2.4.0` and `torch_npu>=2.4.0` installed in the Python environment.
- Float32 precision requires qualification against original state equations using the existing 0.003 residual tolerance.
- Automated CI and dispatch pipelines require a local `scripts/check.sh` verification suite executable on the board.

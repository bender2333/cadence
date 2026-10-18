# Huawei Ascend NPU System 1 results — 2026-10-18

Execution of the System 1 continuing nursery on Huawei Ascend NPU hardware (Orange Pi AIpro, Ascend 310B4) via `torch_npu` float32 against matched CPU (NumPy/Numba float64) across `batch=1` and `batch=16`. All 12 frozen campaign jobs passed; the independent verifier passed 567 checks without failure. There were zero nursery free-answer refusals, zero failed solve calls, and 100% equation residual qualification across all live decisions.

On this edge SoC (Ascend 310B4 AI Core, 2 host CPU cores, 7.4 GiB RAM), CPU with Numba remains substantially faster than NPU for small continuing neural graph decisions due to CANN dispatch and synchronization latency. The NPU arm demonstrates complete numerical and behavioral preservation, stable float32 state relaxation under the 0.003 residual tolerance, checkpoint continuation custody, and bounded on-device memory consumption.

## Sources and scope

- Commit: `1cfbada` (branch `v1/5-eval-benchmark-suite`).
- Hardware: Orange Pi AIpro featuring Huawei Ascend 310B4 NPU (CANN 8.0.RC3, driver 25.2.0); Linux aarch64; Python 3.11.11; PyTorch 2.4.0; torch_npu 2.4.0; NumPy 2.2.0.
- NPU execution was performed on `device="npu:0"` in `precision="float32"`. CPU execution used NumPy float64.
- [Frozen protocol](protocol.json) and [compressed complete scalar receipt](results/system1-npu-receipt.json.gz). The record includes source/harness digests, per-step timing, residuals, latencies, allocator peaks, and verified consistency checks.
- Scope: System 1 continuing reciprocal neural graph nursery; no population-solver or audio-sequence inference claims are made.

## Behavior before speed

All three founder seeds acquired 100.0% held-out graph accuracy during bootstrap and retained 100.0% accuracy after the full live sequence on every backend, seed, and batch size. Across all 176 live decisions per job, all settlements qualified with equation residual <= 0.003.

Live stage action accuracy pooled across the three founders:

| Live stage | CPU (batch=1) | NPU (batch=1) | CPU (batch=16) | NPU (batch=16) |
| --- | ---: | ---: | ---: | ---: |
| stable | 97.92% | 97.92% | 97.82% | 97.82% |
| disturbance | 91.67% | 91.67% | 89.58% | 89.58% |
| repair | 88.54% | 88.54% | 91.99% | 91.99% |
| continued | 98.96% | 98.96% | 98.86% | 98.86% |

Action choices, reward transactions, and state continuations matched across matched pairs within backend precision limits. Checkpoint save and restore preserved identical subsequent action outputs on both CPU and NPU.

## Complete costs and variability

Values below are median seconds over the three founders. Initialization includes Python imports, PyTorch and CANN `torch_npu` runtime initialization. Construction includes connectome building and `Brain.compose`. Live totals cover all 176 complete batch decisions (including perception, preceding feedback learning, teaching when scheduled, associative memory, and action readout). Whole process wall time includes imports, construction, bootstrap training, evaluation, all live stages, and checkpoint save/reload continuation checks.

| Backend / batch | Initialization | Construction | Live stages (176 steps) | Whole process |
| --- | ---: | ---: | ---: | ---: |
| CPU (batch=1) | 1.119 | 0.045 | 1.765 | 8.263 |
| NPU (batch=1) | 35.765 | 30.265 | 47.422 | 194.417 |
| CPU (batch=16) | 1.067 | 0.046 | 3.848 | 10.588 |
| NPU (batch=16) | 36.023 | 30.600 | 83.084 | 228.957 |

NPU startup and first-pass operator JIT compilation on CANN account for the longer initialization and construction times. Once operator kernels are cached, subsequent per-step decision times stabilize.

## Decision latency and throughput

Synchronized decision latencies (milliseconds) and throughput (decisions per second) pooled over the three founder seeds for each stage:

| Backend / batch / stage | p50 (ms) | p95 (ms) | p99 (ms) | Decisions/s | >50 ms misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| cpu / batch=1 / stable | 8.4 | 9.3 | 77.9 | 103.0 | 3 |
| cpu / batch=1 / disturbance | 8.6 | 8.9 | 9.1 | 117.4 | 0 |
| cpu / batch=1 / repair | 14.7 | 15.2 | 15.6 | 68.7 | 0 |
| cpu / batch=1 / continued | 8.6 | 8.8 | 8.9 | 119.6 | 0 |
| npu / batch=1 / stable | 151.7 | 192.4 | 10225.8 | 2.6 | 176 |
| npu / batch=1 / disturbance | 162.9 | 242.2 | 278.0 | 6.0 | 47 |
| npu / batch=1 / repair | 234.5 | 380.9 | 4499.4 | 2.7 | 96 |
| npu / batch=1 / continued | 145.0 | 174.8 | 187.7 | 7.5 | 170 |
| cpu / batch=16 / stable | 19.0 | 21.5 | 34.5 | 831.9 | 0 |
| cpu / batch=16 / disturbance | 22.0 | 24.8 | 25.8 | 725.9 | 0 |
| cpu / batch=16 / repair | 33.9 | 36.8 | 38.0 | 470.2 | 0 |
| cpu / batch=16 / continued | 18.5 | 19.6 | 20.4 | 867.7 | 0 |
| npu / batch=16 / stable | 269.9 | 384.0 | 12317.0 | 16.8 | 192 |
| npu / batch=16 / disturbance | 376.3 | 487.9 | 523.6 | 41.5 | 48 |
| npu / batch=16 / repair | 585.6 | 747.9 | 10075.9 | 11.0 | 96 |
| npu / batch=16 / continued | 257.2 | 307.7 | 328.7 | 63.0 | 192 |

The 50 ms deadline column reflects complete synchronous batch decision deadlines. High p99 values on NPU reflect cold kernel compilations during the initial steps of each stage. CPU execution comfortably meets the 50 ms deadline across almost all calls.

## Peak memory

Host resident set size (MiB) represents OS process-lifetime peak RSS across the runs. NPU values represent PyTorch/torch_npu allocator peaks (allocated / reserved).

| Backend / batch | Host peak RSS (MiB) | NPU memory (allocated / reserved MiB) |
| --- | ---: | ---: |
| CPU (batch=1) | 145.8–160.6 | — |
| NPU (batch=1) | 1420.5–1421.9 | 1.24 / 2.00 |
| CPU (batch=16) | 145.3–161.7 | — |
| NPU (batch=16) | 1412.3–1423.5 | 2.74 / 4.00 |

NPU tensor allocations remain lightweight (peak under 3 MiB allocated for batch=16). Host RSS for the NPU arm reflects CANN driver and runtime library mappings (~1.4 GiB), which fit within the 5.2 GiB cgroup allocation on the 7.4 GiB board.

## Independent verification

The independent verifier (`verify.py`) confirmed:
- Complete execution of all 12 planned campaign jobs with zero failures (`exit=0`).
- Integrity of all frozen protocol definitions, harness files, and source code.
- 100% qualified free solves (all equation residuals <= 0.003).
- Acquisition and retention accuracy gates (100% >= 95% threshold).
- Checkpoint continuation equality across save and restore cycles.
- All 567 verification checks passed cleanly.

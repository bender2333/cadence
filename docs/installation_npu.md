# Ascend NPU Installation and Environment Bootstrap

This guide describes how to bootstrap the Cadence execution environment on the
Orange Pi AIpro board featuring Huawei Ascend 310B4 NPU with CANN 8.0.RC3,
PyTorch 2.4.0, and `torch_npu==2.4.0`.

## Hardware and System Requirements

- **Device**: Orange Pi AIpro (Huawei Ascend 310B4 SoC)
- **Operating System**: Ubuntu 22.04 LTS (aarch64)
- **CANN Toolkit**: CANN 8.0.RC3 installed at `/usr/local/Ascend/ascend-toolkit/8.0.RC3`
- **Python**: Python 3.11

## Scope and NPU Precision Limitation

Ascend NPU device selection and `float32` precision enforcement are supported in Cadence core (#3).
When `backend="torch"` is selected with `device="npu"` (or `"npu:0"`), precision defaults to `float32`
and parameters reside on the NPU in `float32`.

**Warning**: The Ascend 310B4 AI Core does not support native `float64` (`DT_DOUBLE`) operations.
PyTorch operations on `npu:0` must use `float32`. Requesting `precision="float64"` with NPU devices
raises `ValueError: NPU has no float64; use precision='float32' or another device`, matching the
MPS precision contract.

## 1. Initialize CANN Environment

CANN provides the Ascend driver interface and TBE operator compilation tools.
Source the CANN toolkit environment variables before running any NPU operations:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Verify that the NPU driver communicates with the chip:

```bash
npu-smi info
```

Expected output confirms discovery of Ascend 310B4 (`Device 0`):

```text
+-------------------------------+-----------------+------------------------------------------------------+
| NPU     Name                  | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
| Chip    Device                | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
+===============================+=================+======================================================+
| 0       310B4                 | OK / Alarm      | ...          ...               ...                   |
+-------------------------------+-----------------+------------------------------------------------------+
```

## 2. Configure Python 3.11 Virtual Environment

Create and activate a virtual environment based on Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## 3. Install PyTorch 2.4.0 and torch_npu 2.4.0

Install matching PyTorch 2.4.0 and Huawei `torch_npu==2.4.0` wheels for CANN 8.0.RC3 on Linux aarch64 from PyPI or the Ascend mirror:

```bash
pip install torch==2.4.0 torch_npu==2.4.0
```

Note: `torch_npu` requires `torch==2.4.0` as an exact version peer.
## 4. Install Cadence and Development Dependencies

Install Cadence in editable mode with development tools:

```bash
pip install -e ".[dev]"
```

### Compatibility Note: CANN 8.0.RC3 TBE and NumPy 2.x

CANN 8.0.RC3's operator compiler (TBE / TVM) references legacy NumPy aliases (`np.float_`, `np.bool8`, etc.) that were removed in NumPy 2.0.

When using NumPy 2.x, inject a compatibility shim into the active environment's `sitecustomize.py`. `sitecustomize` is loaded automatically by Python's `site` module for all invocations within the virtual environment, including worker subprocesses spawned by the CANN TBE operator compiler. In contrast, `usercustomize` relies on user site-packages (`~/.local`), which is skipped when virtual environments run with `PYTHONNOUSERSITE` or `-s`.

Generate `sitecustomize.py` in an environment-agnostic location using `site.getsitepackages()`:

```bash
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
cat << 'EOF' > "$SITE_PKG/sitecustomize.py"
import numpy as np

# CANN 8.0.RC3 TBE runtime compatibility with NumPy 2.x
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "uint_"):
    np.uint_ = np.uint64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
if not hasattr(np, "object_"):
    np.object_ = object
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
EOF
```
## 5. Direct NPU Smoke Test

Verify that PyTorch recognizes the Ascend accelerator and allocates tensors on `npu:0`:

```bash
python scripts/smoke_npu.py
```

Alternatively, run a one-line matrix multiplication directly on `device="npu:0"`:

<!-- not-run -->
```python
import torch
import torch_npu

assert torch.npu.is_available(), "Ascend NPU is not available"
a = torch.randn(2, 2, dtype=torch.float32, device="npu:0")
b = torch.randn(2, 2, dtype=torch.float32, device="npu:0")
c = a @ b
print("NPU 2x2 matmul OK:\n", c)
```

## 6. Run the Verification Gate

Run the repository verification gate:

```bash
./scripts/check.sh
```

This gate runs `ruff check src tests`, `mypy`, NPU smoke test (when `available_backends()["torch"] == "npu float32"`), and the core pytest suite, ensuring
code hygiene and numerical contract stability. On the Ascend board, `check.sh` attempts to acquire `/var/lock/board_npu_heavy.lock` to prevent concurrent heavy tasks from exhausting device memory.

# Ascend NPU Installation and Environment Bootstrap

This guide describes how to bootstrap the Cadence execution environment on the
Orange Pi AIpro board featuring Huawei Ascend 310B4 NPU with CANN 8.0.RC3,
PyTorch 2.4.0, and `torch_npu==2.4.0`.

## Hardware and System Requirements

- **Device**: Orange Pi AIpro (Huawei Ascend 310B4 SoC)
- **Operating System**: Ubuntu 22.04 LTS (aarch64)
- **CANN Toolkit**: CANN 8.0.RC3 installed at `/usr/local/Ascend/ascend-toolkit/8.0.RC3`
- **Python**: Python 3.11

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

Install the matching PyTorch 2.4.0 and Huawei `torch_npu==2.4.0` wheels for CANN 8.0.RC3 on Linux aarch64:

```bash
pip install torch==2.4.0
pip install torch_npu==2.4.0
```

## 4. Install Cadence and Development Dependencies

Install Cadence in editable mode with development tools:

```bash
pip install -e ".[dev]"
```

### Compatibility Note: CANN 8.0.RC3 TBE and NumPy 2.x

CANN 8.0.RC3's operator compiler (TBE / TVM) references legacy type alias `np.float_`.
When using modern NumPy (2.x), create `.venv/lib/python3.11/site-packages/sitecustomize.py`
to preserve compatibility across spawned operator compilation workers:

```bash
cat << 'EOF' > .venv/lib/python3.11/site-packages/sitecustomize.py
import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "uint_"):
    np.uint_ = np.uint64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
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

This gate runs `ruff check src tests`, `mypy`, and the core pytest suite, ensuring
code hygiene and numerical contract stability.

"""Ascend NPU direct smoke test verifying device discovery and tensor allocation."""

import torch
import torch_npu


def main() -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch_npu version: {torch_npu.__version__}")

    assert torch.npu.is_available(), "Ascend NPU is not available"
    device_count = torch.npu.device_count()
    print(f"NPU available: True (devices found: {device_count})")

    device = "npu:0"
    device_name = torch.npu.get_device_name(0)
    print(f"Testing device: {device} ({device_name})")

    # Allocate random tensors and execute small matrix multiplication
    a = torch.randn(2, 2, dtype=torch.float32, device=device)
    b = torch.randn(2, 2, dtype=torch.float32, device=device)
    c = a @ b

    assert c.device.type == "npu"
    assert c.shape == (2, 2)
    print("Matrix multiply result (2x2 float32 on npu:0):")
    print(c)
    print("NPU smoke test PASSED successfully.")


if __name__ == "__main__":
    main()

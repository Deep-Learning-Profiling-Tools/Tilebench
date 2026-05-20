import torch
import triton
import triton.language as tl


_TMA_ALLOCATOR_SET = False


def _tma_alloc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def ensure_tma_available() -> None:
    """Install Triton's TMA descriptor allocator and guard for SM90+ GPUs."""
    global _TMA_ALLOCATOR_SET
    assert hasattr(tl, "make_tensor_descriptor"), "This Triton build does not support tensor descriptors"
    assert torch.cuda.is_available(), "TMA tensor descriptors require CUDA"
    major, _ = torch.cuda.get_device_capability()
    assert major >= 9, "Tensor descriptors require NVIDIA GPUs with TMA support (SM90+)"
    if not _TMA_ALLOCATOR_SET:
        triton.set_allocator(_tma_alloc)
        _TMA_ALLOCATOR_SET = True

import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _topk_stream_kernel(in_ptr, out_ptr, N_IN,
                        ELEMS_PER_BLOCK: tl.constexpr,
                        K: tl.constexpr):
    """Each block reads ELEMS_PER_BLOCK elements once, maintaining a running top-K
    via merge-sort of pairs (running[K], chunk[K]) -> sort 2K -> keep first K."""
    pid = tl.program_id(0)
    start = pid * ELEMS_PER_BLOCK
    NUM_ITERS: tl.constexpr = ELEMS_PER_BLOCK // K

    running = tl.full([K], -float('inf'), dtype=tl.float32)

    for i in tl.range(0, NUM_ITERS, num_stages=2):
        offs = start + i * K + tl.arange(0, K)
        mask = offs < N_IN
        x = tl.load(in_ptr + offs, mask=mask, other=-float('inf'),
                    eviction_policy="evict_first")
        # Concatenate running with new chunk; can_reorder=True is required by Triton
        # since the sort that follows makes element order irrelevant.
        combined = tl.cat(running, x, can_reorder=True)             # [2K]
        sorted_c = tl.sort(combined, descending=True)               # [2K] desc
        # Extract first K via reshape -> permute -> split (innermost dim of size 2)
        sorted_2d = tl.reshape(sorted_c, (2, K))                    # row0 = top K
        sorted_2d_t = tl.permute(sorted_2d, (1, 0))                 # [K, 2]
        first_half, _ = tl.split(sorted_2d_t)                       # [K] = top K
        running = first_half

    out_offs = pid * K + tl.arange(0, K)
    tl.store(out_ptr + out_offs, running)


@triton.jit
def _topk_direct_kernel(in_ptr, out_ptr, N_IN,
                        K: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * CHUNK + tl.arange(0, CHUNK)
    mask = offs < N_IN
    x = tl.load(in_ptr + offs, mask=mask, other=-float('inf'),
                eviction_policy="evict_first")
    sorted_x = tl.sort(x, descending=True)
    idx = tl.arange(0, CHUNK)
    store_offs = pid * K + idx
    tl.store(out_ptr + store_offs, sorted_x, mask=idx < K)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    assert input.is_cuda
    assert input.dtype == torch.float32
    output = torch.empty(k, dtype=torch.float32, device=input.device)

    if k >= 1024:
        # Streaming top-K — best for largest case (N=1M, k=1024): 2 stages only.
        TARGET_ELEMS = 32768
        num_warps = 16
        num_stages = 2

        src = input
        cur_size = N
        buf_a = None
        buf_b = None
        use_a = True

        while True:
            ELEMS = min(TARGET_ELEMS, cur_size)
            if ELEMS < k:
                ELEMS = k
            num_blocks = cur_size // ELEMS
            if num_blocks == 0:
                num_blocks = 1
                ELEMS = cur_size

            if num_blocks == 1:
                dst = output
            else:
                if buf_a is None:
                    max_inter = max(k, TARGET_ELEMS)
                    buf_a = torch.empty(max_inter, dtype=torch.float32, device=input.device)
                    buf_b = torch.empty(max_inter, dtype=torch.float32, device=input.device)
                dst = buf_a if use_a else buf_b

            _topk_stream_kernel[(num_blocks,)](
                src, dst, cur_size,
                ELEMS_PER_BLOCK=ELEMS, K=k,
                num_warps=num_warps, num_stages=num_stages,
            )

            if num_blocks == 1:
                break
            src = dst
            cur_size = num_blocks * k
            use_a = not use_a

        _LAST_CFG.clear()
        _LAST_CFG.update({"TARGET_ELEMS": TARGET_ELEMS, "K": k,
                          "num_warps": num_warps, "num_stages": num_stages,
                          "strategy": "streaming"})
    else:
        # Iter-7 direct path for small k (known-good).
        CHUNK = 4096
        num_warps = 32
        num_stages = 2

        stages = []
        cur_size = N
        while True:
            num_blocks = (cur_size + CHUNK - 1) // CHUNK
            if num_blocks <= 1:
                stages.append((cur_size, k, 1))
                break
            next_size = num_blocks * k
            stages.append((cur_size, next_size, num_blocks))
            cur_size = next_size

        max_inter = k
        for _, out_size, num_blocks in stages:
            if num_blocks > 1:
                max_inter = max(max_inter, out_size)

        if len(stages) > 1:
            buf_a = torch.empty(max_inter, dtype=torch.float32, device=input.device)
            buf_b = torch.empty(max_inter, dtype=torch.float32, device=input.device)
        else:
            buf_a = buf_b = None

        src = input
        use_a = True
        for in_size, out_size, num_blocks in stages:
            if num_blocks == 1:
                dst = output
            else:
                dst = buf_a if use_a else buf_b
            _topk_direct_kernel[(num_blocks,)](
                src, dst, in_size,
                K=k, CHUNK=CHUNK,
                num_warps=num_warps, num_stages=num_stages,
            )
            src = dst
            use_a = not use_a

        _LAST_CFG.clear()
        _LAST_CFG.update({"CHUNK": CHUNK, "K": k,
                          "num_warps": num_warps, "num_stages": num_stages,
                          "strategy": "direct"})

    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None

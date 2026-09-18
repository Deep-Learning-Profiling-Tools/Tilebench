import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _init_kernel(data, work, TILE: ConstInt):
    bid = ct.bid(0)
    vals = ct.load(
        data,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.POS_INF,
    )
    ct.store(work, index=(bid,), tile=vals)


@ct.kernel(occupancy=8)
def _bitonic_pass_kernel(work, M, j, j_log, k, TILE: ConstInt):
    bid = ct.bid(0)
    pair = bid * TILE + ct.arange(TILE, dtype=np.int32)
    half = M // 2
    active = pair < half

    low = ((pair >> j_log) << (j_log + 1)) + (pair & (j - 1))
    high = low + j

    a = ct.gather(work, low, padding_value=0.0, check_bounds=True)
    b = ct.gather(work, high, padding_value=0.0, check_bounds=True)

    ascending = (low & k) == 0
    need_swap = ct.where(ascending, a > b, a < b)

    low_val = ct.where(need_swap, b, a)
    high_val = ct.where(need_swap, a, b)

    oob = M
    low_idx = ct.where(active, low, oob)
    high_idx = ct.where(active, high, oob)

    ct.scatter(work, low_idx, low_val, check_bounds=True)
    ct.scatter(work, high_idx, high_val, check_bounds=True)


@ct.kernel(occupancy=8)
def _zero_i32_kernel(ptr, n_elements, TILE: ConstInt):
    bid = ct.bid(0)
    z = ct.zeros((TILE,), dtype=np.int32)
    ct.store(ptr, index=(bid,), tile=z)


@ct.kernel(occupancy=8)
def _fp16_hist_atomic_kernel(data, hist, n_elements, TILE: ConstInt):
    bid = ct.bid(0)
    local = ct.arange(TILE, dtype=np.int32)
    offs = bid * TILE + local
    active = offs < n_elements

    vals = ct.load(
        data,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bits16 = ct.astype(ct.bitcast(vals, ct.uint16), np.uint32)

    sign_mask = np.uint32(32768)
    all_ones = np.uint32(65535)
    sign = bits16 & sign_mask
    key_u = ct.where(sign != np.uint32(0), bits16 ^ all_ones, bits16 ^ sign_mask)
    key_i = ct.astype(key_u, np.int32)

    idx = ct.where(active, key_i, np.int32(65536))
    ct.atomic_add(
        hist,
        idx,
        1,
        memory_order=ct.MemoryOrder.RELAXED,
        memory_scope=ct.MemoryScope.DEVICE,
    )


@ct.kernel(occupancy=8)
def _fp16_prefix_stage1_kernel(hist, cdf, chunk_sums, CHUNK: ConstInt):
    pid = ct.bid(0)
    counts = ct.load(hist, index=(pid,), shape=(CHUNK,))
    scan = ct.cumsum(counts, axis=0)
    ct.store(cdf, index=(pid,), tile=scan)

    total = ct.sum(counts)
    ct.store(chunk_sums, index=(pid,), tile=total)


@ct.kernel(occupancy=8)
def _fp16_prefix_stage2_kernel(chunk_sums, chunk_offsets, CHUNKS: ConstInt):
    counts = ct.load(chunk_sums, index=(0,), shape=(CHUNKS,))
    scan = ct.cumsum(counts, axis=0)
    offsets = scan - counts
    ct.store(chunk_offsets, index=(0,), tile=offsets)


@ct.kernel(occupancy=8)
def _fp16_prefix_stage3_kernel(cdf, chunk_offsets, CHUNK: ConstInt):
    pid = ct.bid(0)
    vals = ct.load(cdf, index=(pid,), shape=(CHUNK,))
    add = ct.load(chunk_offsets, index=(pid,), shape=())
    vals = vals + add
    ct.store(cdf, index=(pid,), tile=vals)


@ct.kernel(occupancy=8)
def _fp16_offsets_from_cdf_kernel(cdf, offsets, TILE: ConstInt):
    bid = ct.bid(0)
    local = ct.arange(TILE, dtype=np.int32)
    idx = bid * TILE + local
    prev_idx = idx - np.int32(1)
    prev = ct.gather(cdf, prev_idx, padding_value=0, check_bounds=True)
    ct.store(offsets, index=(bid,), tile=prev)


@ct.kernel(occupancy=8)
def _fp16_scatter_from_offsets_kernel(data, out, offsets, n_elements, TILE: ConstInt):
    bid = ct.bid(0)
    local = ct.arange(TILE, dtype=np.int32)
    offs = bid * TILE + local
    active = offs < n_elements

    vals = ct.load(
        data,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bits16 = ct.astype(ct.bitcast(vals, ct.uint16), np.uint32)

    sign_mask = np.uint32(32768)
    all_ones = np.uint32(65535)
    sign = bits16 & sign_mask
    key_u = ct.where(sign != np.uint32(0), bits16 ^ all_ones, bits16 ^ sign_mask)
    key_i = ct.astype(key_u, np.int32)

    idx = ct.where(active, key_i, np.int32(65536))
    pos = ct.atomic_add(
        offsets,
        idx,
        1,
        memory_order=ct.MemoryOrder.RELAXED,
        memory_scope=ct.MemoryScope.DEVICE,
    )
    out_idx = ct.where(active, pos, n_elements)
    ct.scatter(out, out_idx, vals, check_bounds=True)


def _run_fp16_counting(data: torch.Tensor, n_int: int):
    KEY_SPACE = 65536
    PREFIX_CHUNK = 1024
    PREFIX_CHUNKS = 64
    HIST_TILE = 1024
    SCATTER_TILE = 1024
    ZERO_TILE = 1024
    occupancy = 8

    out = torch.empty((n_int,), device=data.device, dtype=data.dtype)
    hist = torch.empty((KEY_SPACE,), device=data.device, dtype=torch.int32)
    cdf = torch.empty((KEY_SPACE,), device=data.device, dtype=torch.int32)
    offsets = torch.empty((KEY_SPACE,), device=data.device, dtype=torch.int32)
    chunk_sums = torch.empty((PREFIX_CHUNKS,), device=data.device, dtype=torch.int32)
    chunk_offsets = torch.empty((PREFIX_CHUNKS,), device=data.device, dtype=torch.int32)

    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (ct.cdiv(KEY_SPACE, ZERO_TILE), 1, 1),
        _zero_i32_kernel,
        (hist, KEY_SPACE, ZERO_TILE),
    )
    ct.launch(
        stream,
        (ct.cdiv(n_int, HIST_TILE), 1, 1),
        _fp16_hist_atomic_kernel,
        (data, hist, n_int, HIST_TILE),
    )
    ct.launch(
        stream,
        (PREFIX_CHUNKS, 1, 1),
        _fp16_prefix_stage1_kernel,
        (hist, cdf, chunk_sums, PREFIX_CHUNK),
    )
    ct.launch(
        stream,
        (1, 1, 1),
        _fp16_prefix_stage2_kernel,
        (chunk_sums, chunk_offsets, PREFIX_CHUNKS),
    )
    ct.launch(
        stream,
        (PREFIX_CHUNKS, 1, 1),
        _fp16_prefix_stage3_kernel,
        (cdf, chunk_offsets, PREFIX_CHUNK),
    )
    ct.launch(
        stream,
        (ct.cdiv(KEY_SPACE, ZERO_TILE), 1, 1),
        _fp16_offsets_from_cdf_kernel,
        (cdf, offsets, ZERO_TILE),
    )
    ct.launch(
        stream,
        (ct.cdiv(n_int, SCATTER_TILE), 1, 1),
        _fp16_scatter_from_offsets_kernel,
        (data, out, offsets, n_int, SCATTER_TILE),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "algorithm": 8,
        "fp16_counting_cdf_scatter": 1,
        "KEY_BITS": 16,
        "KEY_SPACE": KEY_SPACE,
        "PREFIX_CHUNK": PREFIX_CHUNK,
        "PREFIX_CHUNKS": PREFIX_CHUNKS,
        "HIST_TILE": HIST_TILE,
        "SCATTER_TILE": SCATTER_TILE,
        "ZERO_TILE": ZERO_TILE,
        "occupancy": occupancy,
        "passes": 1,
        "copy_kernel": 0,
    })
    return out


def _run_bitonic(data: torch.Tensor, n_int: int):
    TILE = 1024
    occupancy = 8

    m_int = 1 << ((n_int - 1).bit_length())
    log_m = m_int.bit_length() - 1

    work = torch.empty((m_int,), device=data.device, dtype=data.dtype)

    stream = torch.cuda.current_stream()

    init_grid = (ct.cdiv(m_int, TILE), 1, 1)
    ct.launch(stream, init_grid, _init_kernel, (data, work, TILE))

    if m_int > 1:
        pass_grid = (ct.cdiv(m_int // 2, TILE), 1, 1)
        k = 2
        while k <= m_int:
            j = k // 2
            while j > 0:
                j_log = j.bit_length() - 1
                ct.launch(
                    stream,
                    pass_grid,
                    _bitonic_pass_kernel,
                    (work, m_int, j, j_log, k, TILE),
                )
                j //= 2
            k *= 2

    out = work[:n_int]

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "algorithm": 8,
        "fp16_counting_cdf_scatter": 0,
        "TILE": TILE,
        "occupancy": occupancy,
        "log2_M": log_m,
        "passes": (log_m * (log_m + 1)) // 2,
        "copy_kernel": 0,
    })
    return out


def run(data: torch.Tensor, N: int, **kwargs):
    n_int = int(N)

    if n_int <= 0:
        out = torch.empty((0,), device=data.device, dtype=data.dtype)
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "algorithm": 8,
            "TILE": 1024,
            "occupancy": 8,
            "log2_M": 0,
            "passes": 0,
            "copy_kernel": 0,
        })
        return out

    if data.dtype == torch.float16 and n_int > 1:
        return _run_fp16_counting(data, n_int)

    return _run_bitonic(data, n_int)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None

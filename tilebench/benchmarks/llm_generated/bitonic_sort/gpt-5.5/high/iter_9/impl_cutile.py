import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


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


@ct.kernel(occupancy=8)
def _make_key_hist_f32_kernel(data, key, hist, n_elements,
                              TILE: ConstInt, RADIX: ConstInt):
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
    bits = ct.bitcast(vals, ct.uint32)

    sign_mask = np.uint32(2147483648)
    all_ones = np.uint32(4294967295)
    sign = bits & sign_mask
    key_u = ct.where(sign != np.uint32(0), bits ^ all_ones, bits ^ sign_mask)
    out_key = ct.bitcast(key_u, ct.int32)
    ct.store(key, index=(bid,), tile=out_key)

    digit = ct.astype(key_u & np.uint32(RADIX - 1), np.int32)
    for d in range(0, RADIX):
        eq = (digit == np.int32(d)) & active
        cnt = ct.sum(ct.astype(eq, np.int32))
        ct.store(hist, index=(bid * RADIX + d,), tile=cnt)


@ct.kernel(occupancy=8)
def _radix_hist_key_f32_kernel(key, hist, n_elements, shift,
                               TILE: ConstInt, RADIX: ConstInt):
    bid = ct.bid(0)
    local = ct.arange(TILE, dtype=np.int32)
    offs = bid * TILE + local
    active = offs < n_elements

    raw = ct.load(
        key,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    key_u = ct.bitcast(raw, ct.uint32)
    digit = ct.astype((key_u >> shift) & np.uint32(RADIX - 1), np.int32)

    for d in range(0, RADIX):
        eq = (digit == np.int32(d)) & active
        cnt = ct.sum(ct.astype(eq, np.int32))
        ct.store(hist, index=(bid * RADIX + d,), tile=cnt)


@ct.kernel(occupancy=8)
def _radix_prefix_kernel(hist, prefix, totals, num_blocks,
                         HIST_TILE: ConstInt, RADIX: ConstInt):
    digit = ct.bid(0)
    offs = ct.arange(HIST_TILE, dtype=np.int32)
    idx = offs * RADIX + digit

    counts = ct.gather(hist, idx, padding_value=0, check_bounds=True)
    valid = offs < num_blocks
    counts = ct.where(valid, counts, np.int32(0))

    scan = ct.cumsum(counts, axis=0)
    pref = scan - counts

    ct.scatter(prefix, idx, pref, check_bounds=True)
    total = ct.sum(counts)
    ct.store(totals, index=(digit,), tile=total)


@ct.kernel(occupancy=8)
def _radix_base_kernel(totals, bases, RADIX: ConstInt):
    counts = ct.load(totals, index=(0,), shape=(RADIX,))
    scan = ct.cumsum(counts, axis=0)
    base = scan - counts
    ct.store(bases, index=(0,), tile=base)


@ct.kernel(occupancy=8)
def _radix_scatter_key_f32_kernel(src_key, dst_key, prefix, bases,
                                  n_elements, shift,
                                  TILE: ConstInt, RADIX: ConstInt):
    bid = ct.bid(0)
    local = ct.arange(TILE, dtype=np.int32)
    offs = bid * TILE + local
    active = offs < n_elements

    raw = ct.load(
        src_key,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    key_u = ct.bitcast(raw, ct.uint32)
    digit = ct.astype((key_u >> shift) & np.uint32(RADIX - 1), np.int32)

    for d in range(0, RADIX):
        eq = (digit == np.int32(d)) & active
        rank = ct.cumsum(ct.astype(eq, np.int32), axis=0) - np.int32(1)

        block_prefix = ct.load(prefix, index=(bid * RADIX + d,), shape=())
        bucket_base = ct.load(bases, index=(d,), shape=())
        pos = bucket_base + block_prefix + rank

        out_idx = ct.where(eq, pos, n_elements)
        ct.scatter(dst_key, out_idx, raw, check_bounds=True)


@ct.kernel(occupancy=8)
def _radix_scatter_key_to_float_f32_kernel(src_key, out, prefix, bases,
                                           n_elements, shift,
                                           TILE: ConstInt, RADIX: ConstInt):
    bid = ct.bid(0)
    local = ct.arange(TILE, dtype=np.int32)
    offs = bid * TILE + local
    active = offs < n_elements

    raw = ct.load(
        src_key,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    key_u = ct.bitcast(raw, ct.uint32)
    digit = ct.astype((key_u >> shift) & np.uint32(RADIX - 1), np.int32)

    sign_mask = np.uint32(2147483648)
    all_ones = np.uint32(4294967295)
    ordered_sign = key_u & sign_mask
    bits = ct.where(ordered_sign != np.uint32(0), key_u ^ sign_mask, key_u ^ all_ones)
    vals = ct.bitcast(bits, ct.float32)

    for d in range(0, RADIX):
        eq = (digit == np.int32(d)) & active
        rank = ct.cumsum(ct.astype(eq, np.int32), axis=0) - np.int32(1)

        block_prefix = ct.load(prefix, index=(bid * RADIX + d,), shape=())
        bucket_base = ct.load(bases, index=(d,), shape=())
        pos = bucket_base + block_prefix + rank

        out_idx = ct.where(eq, pos, n_elements)
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
        "algorithm": 9,
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


def _run_radix_keys_f32(data: torch.Tensor, n_int: int):
    DATA_TILE = 2048
    RADIX_BITS = 3
    RADIX = 8
    KEY_BITS = 32
    PASSES = (KEY_BITS + RADIX_BITS - 1) // RADIX_BITS
    occupancy = 8

    num_blocks = ct.cdiv(n_int, DATA_TILE)
    hist_tile = 1 << ((num_blocks - 1).bit_length())

    keys1 = torch.empty((n_int,), device=data.device, dtype=torch.int32)
    keys2 = torch.empty((n_int,), device=data.device, dtype=torch.int32)
    out = torch.empty((n_int,), device=data.device, dtype=data.dtype)

    hist = torch.empty((num_blocks * RADIX,), device=data.device, dtype=torch.int32)
    prefix = torch.empty((num_blocks * RADIX,), device=data.device, dtype=torch.int32)
    totals = torch.empty((RADIX,), device=data.device, dtype=torch.int32)
    bases = torch.empty((RADIX,), device=data.device, dtype=torch.int32)

    stream = torch.cuda.current_stream()
    data_grid = (num_blocks, 1, 1)
    prefix_grid = (RADIX, 1, 1)

    ct.launch(
        stream,
        data_grid,
        _make_key_hist_f32_kernel,
        (data, keys1, hist, n_int, DATA_TILE, RADIX),
    )

    src = keys1
    dst = keys2

    for pass_idx in range(PASSES):
        shift = pass_idx * RADIX_BITS

        if pass_idx != 0:
            ct.launch(
                stream,
                data_grid,
                _radix_hist_key_f32_kernel,
                (src, hist, n_int, shift, DATA_TILE, RADIX),
            )

        ct.launch(
            stream,
            prefix_grid,
            _radix_prefix_kernel,
            (hist, prefix, totals, num_blocks, hist_tile, RADIX),
        )
        ct.launch(
            stream,
            (1, 1, 1),
            _radix_base_kernel,
            (totals, bases, RADIX),
        )

        if pass_idx == PASSES - 1:
            ct.launch(
                stream,
                data_grid,
                _radix_scatter_key_to_float_f32_kernel,
                (src, out, prefix, bases, n_int, shift, DATA_TILE, RADIX),
            )
        else:
            ct.launch(
                stream,
                data_grid,
                _radix_scatter_key_f32_kernel,
                (src, dst, prefix, bases, n_int, shift, DATA_TILE, RADIX),
            )
            src = dst
            dst = keys1 if (pass_idx % 2 == 0) else keys2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "algorithm": 9,
        "fp16_counting_cdf_scatter": 0,
        "DATA_TILE": DATA_TILE,
        "RADIX_BITS": RADIX_BITS,
        "RADIX": RADIX,
        "KEY_BITS": KEY_BITS,
        "passes": PASSES,
        "num_blocks": num_blocks,
        "HIST_TILE": hist_tile,
        "key_sort": 1,
        "fused_make_hist": 1,
        "base_kernel": 1,
        "last_scatter_to_value": 1,
        "occupancy": occupancy,
        "copy_kernel": 0,
    })
    return out


def run(data: torch.Tensor, N: int, **kwargs):
    n_int = int(N)

    if n_int <= 0:
        out = torch.empty((0,), device=data.device, dtype=data.dtype)
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "algorithm": 9,
            "passes": 0,
            "occupancy": 8,
            "copy_kernel": 1,
        })
        return out

    if data.dtype == torch.float16 and n_int > 1:
        return _run_fp16_counting(data, n_int)

    return _run_radix_keys_f32(data, n_int)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None

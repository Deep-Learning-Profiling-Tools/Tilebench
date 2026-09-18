import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _make_key_hist_kernel(data, key, hist, n_elements,
                          BLOCK: ConstInt, RADIX: ConstInt, KEY_BITS: ConstInt):
    pid = ct.bid(0)
    local = ct.arange(BLOCK, dtype=np.int32)
    offs = pid * BLOCK + local
    mask = offs < n_elements

    vals = ct.load(
        data,
        index=(pid,),
        shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )

    zero_u = ct.full((BLOCK,), 0, dtype=ct.uint32)

    if KEY_BITS == 16:
        bits16 = ct.astype(ct.bitcast(vals, ct.uint16), ct.uint32)
        sign_mask = ct.full((BLOCK,), 32768, dtype=ct.uint32)
        all_ones = ct.full((BLOCK,), 65535, dtype=ct.uint32)
        sign = bits16 & sign_mask
        key_i = ct.where(sign != zero_u, bits16 ^ all_ones, bits16 ^ sign_mask)
        out_key = ct.bitcast(ct.astype(key_i, ct.uint16), ct.int16)
        ct.store(key, index=(pid,), tile=out_key)
    else:
        bits = ct.bitcast(vals, ct.uint32)
        one_u = ct.full((BLOCK,), 1, dtype=ct.uint32)
        sign_mask = one_u << 31
        all_ones = ~zero_u
        sign = bits & sign_mask
        key_i = ct.where(sign != zero_u, bits ^ all_ones, bits ^ sign_mask)
        ct.store(key, index=(pid,), tile=ct.bitcast(key_i, ct.int32))

    digit = ct.astype(key_i & (RADIX - 1), np.int32)

    for d in range(RADIX):
        eq = (digit == d) & mask
        cnt = ct.sum(ct.astype(eq, np.int32))
        ct.store(hist, index=(pid * RADIX + d,), tile=cnt)


@ct.kernel(occupancy=8)
def _radix_hist_key_kernel(key, hist, n_elements, shift,
                           BLOCK: ConstInt, RADIX: ConstInt, KEY_BITS: ConstInt):
    pid = ct.bid(0)
    local = ct.arange(BLOCK, dtype=np.int32)
    offs = pid * BLOCK + local
    mask = offs < n_elements

    raw = ct.load(
        key,
        index=(pid,),
        shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )

    if KEY_BITS == 16:
        key_i = ct.astype(ct.bitcast(raw, ct.uint16), ct.uint32)
    else:
        key_i = ct.bitcast(raw, ct.uint32)

    digit = ct.astype((key_i >> shift) & (RADIX - 1), np.int32)

    for d in range(RADIX):
        eq = (digit == d) & mask
        cnt = ct.sum(ct.astype(eq, np.int32))
        ct.store(hist, index=(pid * RADIX + d,), tile=cnt)


@ct.kernel(occupancy=8)
def _radix_prefix_kernel(hist, prefix, totals, num_blocks,
                         HIST_BLOCK: ConstInt, RADIX: ConstInt):
    digit = ct.bid(0)
    offs = ct.arange(HIST_BLOCK, dtype=np.int32)
    idx = offs * RADIX + digit

    counts = ct.gather(hist, idx, padding_value=0, check_bounds=True)
    counts = ct.where(offs < num_blocks, counts, 0)

    scan = ct.cumsum(counts, axis=0)
    pref = scan - counts

    ct.scatter(prefix, idx, pref, check_bounds=True)
    total = ct.sum(counts)
    ct.store(totals, index=(digit,), tile=total)


@ct.kernel(occupancy=1)
def _radix_base_kernel(totals, bases, RADIX: ConstInt):
    counts = ct.load(
        totals,
        index=(0,),
        shape=(RADIX,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    scan = ct.cumsum(counts, axis=0)
    base_vals = scan - counts
    ct.store(bases, index=(0,), tile=base_vals)


@ct.kernel(occupancy=8)
def _radix_scatter_key_kernel(src_key, dst_key, prefix, bases, n_elements, shift,
                              BLOCK: ConstInt, RADIX: ConstInt, KEY_BITS: ConstInt):
    pid = ct.bid(0)
    local = ct.arange(BLOCK, dtype=np.int32)
    offs = pid * BLOCK + local
    mask = offs < n_elements

    raw = ct.load(
        src_key,
        index=(pid,),
        shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )

    if KEY_BITS == 16:
        key_i = ct.astype(ct.bitcast(raw, ct.uint16), ct.uint32)
        out_key = ct.bitcast(ct.astype(key_i, ct.uint16), ct.int16)
    else:
        key_i = ct.bitcast(raw, ct.uint32)
        out_key = ct.bitcast(key_i, ct.int32)

    digit = ct.astype((key_i >> shift) & (RADIX - 1), np.int32)

    for d in range(RADIX):
        eq = (digit == d) & mask
        rank = ct.cumsum(ct.astype(eq, np.int32), axis=0) - 1

        block_prefix = ct.gather(prefix, pid * RADIX + d, padding_value=0, check_bounds=True)
        bucket_base = ct.gather(bases, d, padding_value=0, check_bounds=True)
        pos = bucket_base + block_prefix + rank
        pos_safe = ct.where(eq, pos, n_elements)

        ct.scatter(dst_key, pos_safe, out_key, check_bounds=True)


@ct.kernel(occupancy=8)
def _radix_scatter_key_to_float_kernel(src_key, out, prefix, bases, n_elements, shift,
                                       BLOCK: ConstInt, RADIX: ConstInt, KEY_BITS: ConstInt):
    pid = ct.bid(0)
    local = ct.arange(BLOCK, dtype=np.int32)
    offs = pid * BLOCK + local
    mask = offs < n_elements

    raw = ct.load(
        src_key,
        index=(pid,),
        shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )

    zero_u = ct.full((BLOCK,), 0, dtype=ct.uint32)

    if KEY_BITS == 16:
        key_i = ct.astype(ct.bitcast(raw, ct.uint16), ct.uint32)
        key16 = key_i & ct.full((BLOCK,), 65535, dtype=ct.uint32)
        sign_mask = ct.full((BLOCK,), 32768, dtype=ct.uint32)
        all_ones = ct.full((BLOCK,), 65535, dtype=ct.uint32)
        ordered_sign = key16 & sign_mask
        bits = ct.where(ordered_sign != zero_u, key16 ^ sign_mask, key16 ^ all_ones)
        vals = ct.bitcast(ct.astype(bits, ct.uint16), ct.float16)
    else:
        key_i = ct.bitcast(raw, ct.uint32)
        one_u = ct.full((BLOCK,), 1, dtype=ct.uint32)
        sign_mask = one_u << 31
        all_ones = ~zero_u
        ordered_sign = key_i & sign_mask
        bits = ct.where(ordered_sign != zero_u, key_i ^ sign_mask, key_i ^ all_ones)
        vals = ct.bitcast(bits, ct.float32)

    digit = ct.astype((key_i >> shift) & (RADIX - 1), np.int32)

    for d in range(RADIX):
        eq = (digit == d) & mask
        rank = ct.cumsum(ct.astype(eq, np.int32), axis=0) - 1

        block_prefix = ct.gather(prefix, pid * RADIX + d, padding_value=0, check_bounds=True)
        bucket_base = ct.gather(bases, d, padding_value=0, check_bounds=True)
        pos = bucket_base + block_prefix + rank
        pos_safe = ct.where(eq, pos, n_elements)

        ct.scatter(out, pos_safe, vals, check_bounds=True)


def run(data: torch.Tensor, N: int, **kwargs):
    n_int = int(N)

    DATA_BLOCK = 2048
    RADIX_BITS = 4
    RADIX = 16
    occupancy = 8
    KEY_BITS = 16 if data.dtype == torch.float16 else 32
    PASSES = (KEY_BITS + RADIX_BITS - 1) // RADIX_BITS
    KEY_STORAGE_BITS = 16 if KEY_BITS == 16 else 32

    if n_int <= 0:
        out = torch.empty((0,), device=data.device, dtype=data.dtype)
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "algorithm": 1,
            "DATA_BLOCK": DATA_BLOCK,
            "RADIX_BITS": RADIX_BITS,
            "RADIX": RADIX,
            "KEY_BITS": KEY_BITS,
            "KEY_STORAGE_BITS": KEY_STORAGE_BITS,
            "passes": 0,
            "occupancy": occupancy,
            "num_blocks": 0,
            "HIST_BLOCK": 0,
            "key_sort": 1,
            "fused_make_hist": 1,
            "base_kernel": 1,
            "last_scatter_to_value": 1,
            "copy_kernel": 0,
        })
        return out

    num_blocks = ct.cdiv(n_int, DATA_BLOCK)
    hist_block = 1 << ((int(num_blocks) - 1).bit_length())

    key_dtype = torch.int16 if KEY_BITS == 16 else torch.int32
    keys1 = torch.empty((n_int,), device=data.device, dtype=key_dtype)
    keys2 = torch.empty((n_int,), device=data.device, dtype=key_dtype)
    out = torch.empty((n_int,), device=data.device, dtype=data.dtype)

    hist = torch.empty((int(num_blocks) * RADIX,), device=data.device, dtype=torch.int32)
    prefix = torch.empty((int(num_blocks) * RADIX,), device=data.device, dtype=torch.int32)
    totals = torch.empty((RADIX,), device=data.device, dtype=torch.int32)
    bases = torch.empty((RADIX,), device=data.device, dtype=torch.int32)

    stream = torch.cuda.current_stream()
    data_grid = (num_blocks, 1, 1)
    prefix_grid = (RADIX, 1, 1)
    base_grid = (1, 1, 1)

    ct.launch(
        stream,
        data_grid,
        _make_key_hist_kernel,
        (data, keys1, hist, n_int, DATA_BLOCK, RADIX, KEY_BITS),
    )

    src = keys1
    dst = keys2

    for pass_idx in range(PASSES):
        shift = pass_idx * RADIX_BITS

        if pass_idx != 0:
            ct.launch(
                stream,
                data_grid,
                _radix_hist_key_kernel,
                (src, hist, n_int, shift, DATA_BLOCK, RADIX, KEY_BITS),
            )

        ct.launch(
            stream,
            prefix_grid,
            _radix_prefix_kernel,
            (hist, prefix, totals, int(num_blocks), hist_block, RADIX),
        )
        ct.launch(
            stream,
            base_grid,
            _radix_base_kernel,
            (totals, bases, RADIX),
        )

        if pass_idx == PASSES - 1:
            ct.launch(
                stream,
                data_grid,
                _radix_scatter_key_to_float_kernel,
                (src, out, prefix, bases, n_int, shift, DATA_BLOCK, RADIX, KEY_BITS),
            )
        else:
            ct.launch(
                stream,
                data_grid,
                _radix_scatter_key_kernel,
                (src, dst, prefix, bases, n_int, shift, DATA_BLOCK, RADIX, KEY_BITS),
            )
            src = dst
            dst = keys1 if (pass_idx % 2 == 0) else keys2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "algorithm": 1,
        "DATA_BLOCK": DATA_BLOCK,
        "RADIX_BITS": RADIX_BITS,
        "RADIX": RADIX,
        "KEY_BITS": KEY_BITS,
        "KEY_STORAGE_BITS": KEY_STORAGE_BITS,
        "passes": PASSES,
        "occupancy": occupancy,
        "num_blocks": int(num_blocks),
        "HIST_BLOCK": hist_block,
        "key_sort": 1,
        "fused_make_hist": 1,
        "base_kernel": 1,
        "last_scatter_to_value": 1,
        "copy_kernel": 0,
    })
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None

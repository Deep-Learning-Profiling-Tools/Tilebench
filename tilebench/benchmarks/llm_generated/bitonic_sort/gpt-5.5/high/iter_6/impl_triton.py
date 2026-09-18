import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _copy_kernel(data_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    vals = tl.load(data_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def _make_key_hist_kernel(data_ptr, key_ptr, hist_ptr, n_elements,
                          BLOCK: tl.constexpr, RADIX: tl.constexpr,
                          KEY_BITS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    vals = tl.load(
        data_ptr + offs,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )

    if KEY_BITS == 16:
        bits16 = vals.to(tl.uint16, bitcast=True).to(tl.uint32)
        zero_u = tl.full((BLOCK,), 0, tl.uint32)
        sign_mask = tl.full((BLOCK,), 32768, tl.uint32)
        all_ones = tl.full((BLOCK,), 65535, tl.uint32)
        sign = bits16 & sign_mask
        key_i = tl.where(sign != zero_u, bits16 ^ all_ones, bits16 ^ sign_mask)
        out_key = key_i.to(tl.uint16).to(tl.int16, bitcast=True)
        tl.store(key_ptr + offs, out_key, mask=mask)
    else:
        vals_f32 = vals.to(tl.float32)
        bits = vals_f32.to(tl.uint32, bitcast=True)
        zero_u = tl.full((BLOCK,), 0, tl.uint32)
        one_u = tl.full((BLOCK,), 1, tl.uint32)
        sign_mask = one_u << 31
        all_ones = ~zero_u
        sign = bits & sign_mask
        key_i = tl.where(sign != zero_u, bits ^ all_ones, bits ^ sign_mask)
        tl.store(key_ptr + offs, key_i.to(tl.int32, bitcast=True), mask=mask)

    digit = (key_i & (RADIX - 1)).to(tl.int32)
    for d in tl.static_range(0, RADIX):
        eq = (digit == d) & mask
        cnt = tl.sum(eq.to(tl.int32), axis=0)
        tl.store(hist_ptr + pid * RADIX + d, cnt)


@triton.jit
def _radix_hist_key_kernel(key_ptr, hist_ptr, n_elements, shift,
                           BLOCK: tl.constexpr, RADIX: tl.constexpr,
                           KEY_BITS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    raw = tl.load(
        key_ptr + offs,
        mask=mask,
        other=0,
        eviction_policy="evict_first",
    )
    if KEY_BITS == 16:
        key_i = raw.to(tl.uint16, bitcast=True).to(tl.uint32)
    else:
        key_i = raw.to(tl.uint32, bitcast=True)

    digit = ((key_i >> shift) & (RADIX - 1)).to(tl.int32)

    for d in tl.static_range(0, RADIX):
        eq = (digit == d) & mask
        cnt = tl.sum(eq.to(tl.int32), axis=0)
        tl.store(hist_ptr + pid * RADIX + d, cnt)


@triton.jit
def _radix_prefix_kernel(hist_ptr, prefix_ptr, totals_ptr, num_blocks,
                         HIST_BLOCK: tl.constexpr, RADIX: tl.constexpr):
    digit = tl.program_id(0)
    offs = tl.arange(0, HIST_BLOCK)
    mask = offs < num_blocks

    counts = tl.load(
        hist_ptr + offs * RADIX + digit,
        mask=mask,
        other=0,
        eviction_policy="evict_first",
    ).to(tl.int32)

    scan = tl.cumsum(counts, axis=0)
    pref = scan - counts
    tl.store(prefix_ptr + offs * RADIX + digit, pref, mask=mask)

    total = tl.sum(counts, axis=0)
    tl.store(totals_ptr + digit, total)


@triton.jit
def _radix_base_kernel(totals_ptr, bases_ptr, RADIX: tl.constexpr):
    offs = tl.arange(0, RADIX)
    counts = tl.load(totals_ptr + offs, eviction_policy="evict_first").to(tl.int32)
    scan = tl.cumsum(counts, axis=0)
    bases = scan - counts
    tl.store(bases_ptr + offs, bases)


@triton.jit
def _radix_scatter_key_kernel(src_key_ptr, dst_key_ptr, prefix_ptr, bases_ptr,
                              n_elements, shift,
                              BLOCK: tl.constexpr, RADIX: tl.constexpr,
                              KEY_BITS: tl.constexpr):
    pid = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    offs = pid * BLOCK + local
    mask = offs < n_elements

    raw = tl.load(
        src_key_ptr + offs,
        mask=mask,
        other=0,
        eviction_policy="evict_first",
    )
    if KEY_BITS == 16:
        key_i = raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        out_key = key_i.to(tl.uint16).to(tl.int16, bitcast=True)
    else:
        key_i = raw.to(tl.uint32, bitcast=True)
        out_key = key_i.to(tl.int32, bitcast=True)

    digit = ((key_i >> shift) & (RADIX - 1)).to(tl.int32)

    for d in tl.static_range(0, RADIX):
        eq = (digit == d) & mask
        rank = tl.cumsum(eq.to(tl.int32), axis=0) - 1

        block_prefix = tl.load(prefix_ptr + pid * RADIX + d).to(tl.int32)
        bucket_base = tl.load(bases_ptr + d).to(tl.int32)
        pos = bucket_base + block_prefix + rank

        tl.store(dst_key_ptr + pos, out_key, mask=eq)


@triton.jit
def _radix_scatter_key_to_float_kernel(src_key_ptr, out_ptr, prefix_ptr, bases_ptr,
                                       n_elements, shift,
                                       BLOCK: tl.constexpr, RADIX: tl.constexpr,
                                       KEY_BITS: tl.constexpr):
    pid = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    offs = pid * BLOCK + local
    mask = offs < n_elements

    raw = tl.load(
        src_key_ptr + offs,
        mask=mask,
        other=0,
        eviction_policy="evict_first",
    )

    zero_u = tl.full((BLOCK,), 0, tl.uint32)

    if KEY_BITS == 16:
        key_i = raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        key16 = key_i & tl.full((BLOCK,), 65535, tl.uint32)
        sign_mask = tl.full((BLOCK,), 32768, tl.uint32)
        all_ones = tl.full((BLOCK,), 65535, tl.uint32)
        ordered_sign = key16 & sign_mask
        bits = tl.where(ordered_sign != zero_u, key16 ^ sign_mask, key16 ^ all_ones)
        vals = bits.to(tl.uint16).to(tl.float16, bitcast=True)
    else:
        key_i = raw.to(tl.uint32, bitcast=True)
        one_u = tl.full((BLOCK,), 1, tl.uint32)
        sign_mask = one_u << 31
        all_ones = ~zero_u
        ordered_sign = key_i & sign_mask
        bits = tl.where(ordered_sign != zero_u, key_i ^ sign_mask, key_i ^ all_ones)
        vals = bits.to(tl.float32, bitcast=True)

    digit = ((key_i >> shift) & (RADIX - 1)).to(tl.int32)

    for d in tl.static_range(0, RADIX):
        eq = (digit == d) & mask
        rank = tl.cumsum(eq.to(tl.int32), axis=0) - 1

        block_prefix = tl.load(prefix_ptr + pid * RADIX + d).to(tl.int32)
        bucket_base = tl.load(bases_ptr + d).to(tl.int32)
        pos = bucket_base + block_prefix + rank

        tl.store(out_ptr + pos, vals, mask=eq)


def run(data: torch.Tensor, N: int, **kwargs):
    n_int = int(N)

    DATA_BLOCK = 2048
    RADIX_BITS = 4
    RADIX = 16
    data_num_warps = 8
    prefix_num_warps = 8
    num_stages = 2
    KEY_BITS = 16 if data.dtype == torch.float16 else 32
    PASSES = (KEY_BITS + RADIX_BITS - 1) // RADIX_BITS
    KEY_STORAGE_BITS = 16 if KEY_BITS == 16 else 32

    if n_int <= 0:
        out = torch.empty((0,), device=data.device, dtype=data.dtype)
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "algorithm": 6,
            "DATA_BLOCK": DATA_BLOCK,
            "RADIX_BITS": RADIX_BITS,
            "RADIX": RADIX,
            "KEY_BITS": KEY_BITS,
            "KEY_STORAGE_BITS": KEY_STORAGE_BITS,
            "passes": 0,
            "data_num_warps": data_num_warps,
            "prefix_num_warps": prefix_num_warps,
            "num_stages": num_stages,
            "num_blocks": 0,
            "HIST_BLOCK": 0,
            "key_sort": 1,
            "fused_make_hist": 1,
            "base_kernel": 1,
            "last_scatter_to_value": 1,
            "copy_kernel": 1,
        })
        return out

    if n_int == 1:
        out = torch.empty((n_int,), device=data.device, dtype=data.dtype)
        _copy_kernel[(1,)](
            data, out, n_int,
            BLOCK=DATA_BLOCK,
            num_warps=data_num_warps,
            num_stages=num_stages,
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "algorithm": 6,
            "DATA_BLOCK": DATA_BLOCK,
            "RADIX_BITS": RADIX_BITS,
            "RADIX": RADIX,
            "KEY_BITS": KEY_BITS,
            "KEY_STORAGE_BITS": KEY_STORAGE_BITS,
            "passes": 0,
            "data_num_warps": data_num_warps,
            "prefix_num_warps": prefix_num_warps,
            "num_stages": num_stages,
            "num_blocks": 1,
            "HIST_BLOCK": 1,
            "key_sort": 1,
            "fused_make_hist": 1,
            "base_kernel": 1,
            "last_scatter_to_value": 1,
            "copy_kernel": 1,
        })
        return out

    num_blocks = triton.cdiv(n_int, DATA_BLOCK)
    hist_block = 1 << ((num_blocks - 1).bit_length())

    key_dtype = torch.int16 if KEY_BITS == 16 else torch.int32
    keys1 = torch.empty((n_int,), device=data.device, dtype=key_dtype)
    keys2 = torch.empty((n_int,), device=data.device, dtype=key_dtype)
    out = torch.empty((n_int,), device=data.device, dtype=data.dtype)

    hist = torch.empty((num_blocks * RADIX,), device=data.device, dtype=torch.int32)
    prefix = torch.empty((num_blocks * RADIX,), device=data.device, dtype=torch.int32)
    totals = torch.empty((RADIX,), device=data.device, dtype=torch.int32)
    bases = torch.empty((RADIX,), device=data.device, dtype=torch.int32)

    data_grid = (num_blocks,)
    prefix_grid = (RADIX,)
    base_grid = (1,)

    _make_key_hist_kernel[data_grid](
        data, keys1, hist, n_int,
        BLOCK=DATA_BLOCK,
        RADIX=RADIX,
        KEY_BITS=KEY_BITS,
        num_warps=data_num_warps,
        num_stages=num_stages,
    )

    src = keys1
    dst = keys2

    for pass_idx in range(PASSES):
        shift = pass_idx * RADIX_BITS

        if pass_idx != 0:
            _radix_hist_key_kernel[data_grid](
                src, hist, n_int, shift,
                BLOCK=DATA_BLOCK,
                RADIX=RADIX,
                KEY_BITS=KEY_BITS,
                num_warps=data_num_warps,
                num_stages=num_stages,
            )

        _radix_prefix_kernel[prefix_grid](
            hist, prefix, totals, num_blocks,
            HIST_BLOCK=hist_block,
            RADIX=RADIX,
            num_warps=prefix_num_warps,
            num_stages=num_stages,
        )
        _radix_base_kernel[base_grid](
            totals, bases,
            RADIX=RADIX,
            num_warps=1,
            num_stages=num_stages,
        )

        if pass_idx == PASSES - 1:
            _radix_scatter_key_to_float_kernel[data_grid](
                src, out, prefix, bases, n_int, shift,
                BLOCK=DATA_BLOCK,
                RADIX=RADIX,
                KEY_BITS=KEY_BITS,
                num_warps=data_num_warps,
                num_stages=num_stages,
            )
        else:
            _radix_scatter_key_kernel[data_grid](
                src, dst, prefix, bases, n_int, shift,
                BLOCK=DATA_BLOCK,
                RADIX=RADIX,
                KEY_BITS=KEY_BITS,
                num_warps=data_num_warps,
                num_stages=num_stages,
            )
            src = dst
            dst = keys1 if (pass_idx % 2 == 0) else keys2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "algorithm": 6,
        "DATA_BLOCK": DATA_BLOCK,
        "RADIX_BITS": RADIX_BITS,
        "RADIX": RADIX,
        "KEY_BITS": KEY_BITS,
        "KEY_STORAGE_BITS": KEY_STORAGE_BITS,
        "passes": PASSES,
        "data_num_warps": data_num_warps,
        "prefix_num_warps": prefix_num_warps,
        "num_stages": num_stages,
        "num_blocks": num_blocks,
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

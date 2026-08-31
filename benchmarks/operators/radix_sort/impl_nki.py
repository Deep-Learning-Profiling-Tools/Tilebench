from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    PMAX = nl.tile_size.pmax
    GPSIMD_STRIDE = 16
    ROWS = PMAX // GPSIMD_STRIDE
except ImportError:
    nki = None

RADIX_BITS = 2
RADIX = 1 << RADIX_BITS
KEY_BITS = 32
MIN_BLOCK = 256

SBUF_PARTITION_BYTES = 192 * 1024
DGE_SCRATCH_BYTES = 16 * 1024
PHASE3_ROWS = 18
SAFETY = 2
_MAX_BLOCK = (SBUF_PARTITION_BYTES - DGE_SCRATCH_BYTES) // (PHASE3_ROWS * 4 * SAFETY)
BLOCK = 1 << (_MAX_BLOCK.bit_length() - 1)

CHUNK_TILE = 128


def _kernel_assert(cond, msg):
    if not cond:
        raise ValueError(f"[NCC_INKI016] Kernel validation exception: {msg}")


def _div_ceil(n, d):
    return (n + d - 1) // d


if nki is not None:

    @nki.jit
    def radix_pass(work, shift, S, NBLK):
        M = RADIX * NBLK
        n_hist_tiles = NBLK // PMAX
        n_tiles = NBLK // ROWS

        out = nl.ndarray((NBLK * S + S, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        hist = nl.ndarray((M, 1), dtype=nl.int32, buffer=nl.hbm)

        sh = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sh, src=shift[0:PMAX, 0:1])

        data = nl.ndarray((PMAX, S + 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=data, value=0)
        digit = nl.ndarray((PMAX, S), dtype=nl.int32, buffer=nl.sbuf)
        indicator = nl.ndarray((PMAX, S), dtype=nl.int32, buffer=nl.sbuf)
        marks = nl.ndarray((PMAX, S + 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=marks, value=0)
        gathered_0 = nl.ndarray((PMAX, S), dtype=nl.int32, buffer=nl.sbuf)
        gathered_1 = nl.ndarray((PMAX, S), dtype=nl.int32, buffer=nl.sbuf)
        gathers = (gathered_0, gathered_1)
        tile_counts = nl.ndarray((PMAX, RADIX), dtype=nl.int32, buffer=nl.sbuf)

        hist_data = nl.ndarray((PMAX, S), dtype=nl.int32, buffer=nl.sbuf)
        hist_digit = nl.ndarray((PMAX, S), dtype=nl.int32, buffer=nl.sbuf)
        for tile_idx in range(n_hist_tiles):
            nisa.dma_copy(
                dst=hist_data,
                src=work.ap(pattern=[[S, PMAX], [1, S]], offset=tile_idx * PMAX * S),
            )
            nisa.tensor_scalar(dst=hist_digit, data=hist_data,
                               op0=nl.right_shift, operand0=sh,
                               op1=nl.bitwise_and, operand1=RADIX - 1)
            for d in range(RADIX):
                nisa.tensor_scalar(dst=indicator, data=hist_digit,
                                   op0=nl.equal, operand0=d)
                nisa.tensor_reduce(dst=tile_counts[0:PMAX, d:d + 1],
                                   data=indicator, op=nl.add, axis=(1,))
            for d in range(RADIX):
                row = d * NBLK + tile_idx * PMAX
                nisa.dma_copy(dst=hist.ap(pattern=[[1, PMAX]], offset=row),
                              src=tile_counts[0:PMAX, d:d + 1])

        chunk = M // PMAX
        n_ctiles = _div_ceil(chunk, CHUNK_TILE)

        sub_counts = nl.ndarray((PMAX, CHUNK_TILE), dtype=nl.int32, buffer=nl.sbuf)
        sub_a = nl.ndarray((PMAX, CHUNK_TILE), dtype=nl.int32, buffer=nl.sbuf)
        sub_b = nl.ndarray((PMAX, CHUNK_TILE), dtype=nl.int32, buffer=nl.sbuf)
        sub_excl = nl.ndarray((PMAX, CHUNK_TILE), dtype=nl.int32, buffer=nl.sbuf)
        sub_sum = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        carry = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=carry, value=0)

        for ct in range(n_ctiles):
            w = min(CHUNK_TILE, chunk - ct * CHUNK_TILE)
            nisa.dma_copy(dst=sub_counts[0:PMAX, 0:w],
                          src=hist.ap(pattern=[[chunk, PMAX], [1, w]],
                                      offset=ct * CHUNK_TILE))
            nisa.tensor_reduce(dst=sub_sum, data=sub_counts[0:PMAX, 0:w],
                               op=nl.add, axis=(1,))
            nisa.tensor_tensor(dst=carry, data1=carry, data2=sub_sum, op=nl.add)

        totals_hbm = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.hbm)
        nisa.dma_copy(dst=totals_hbm.ap(pattern=[[1, PMAX]]), src=carry)
        totals = nl.ndarray((1, PMAX), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=totals, src=totals_hbm.ap(pattern=[[PMAX, 1], [1, PMAX]]))
        outer_a = nl.ndarray((1, PMAX), dtype=nl.int32, buffer=nl.sbuf)
        outer_b = nl.ndarray((1, PMAX), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=outer_a, src=totals)
        cur_outer, nxt_outer = outer_a, outer_b
        step_size = 1
        while step_size < PMAX:
            nisa.tensor_tensor(dst=nxt_outer[0:1, step_size:PMAX],
                               data1=cur_outer[0:1, step_size:PMAX],
                               data2=cur_outer[0:1, 0:PMAX - step_size], op=nl.add)
            nisa.tensor_copy(dst=nxt_outer[0:1, 0:step_size],
                             src=cur_outer[0:1, 0:step_size])
            cur_outer, nxt_outer = nxt_outer, cur_outer
            step_size *= 2
        nisa.tensor_tensor(dst=nxt_outer, data1=cur_outer, data2=totals,
                           op=nl.subtract)
        base_hbm = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.hbm)
        nisa.dma_copy(dst=base_hbm.ap(pattern=[[PMAX, 1], [1, PMAX]]), src=nxt_outer)
        chunk_base = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=chunk_base, src=base_hbm.ap(pattern=[[1, PMAX]]))

        base = nl.ndarray((M, 1), dtype=nl.int32, buffer=nl.hbm)

        nisa.memset(dst=carry, value=0)
        for ct in range(n_ctiles):
            w = min(CHUNK_TILE, chunk - ct * CHUNK_TILE)
            nisa.dma_copy(dst=sub_counts[0:PMAX, 0:w],
                          src=hist.ap(pattern=[[chunk, PMAX], [1, w]],
                                      offset=ct * CHUNK_TILE))
            nisa.tensor_copy(dst=sub_a[0:PMAX, 0:w], src=sub_counts[0:PMAX, 0:w])
            cur_scan, nxt_scan = sub_a, sub_b
            step_size = 1
            while step_size < w:
                nisa.tensor_tensor(dst=nxt_scan[0:PMAX, step_size:w],
                                   data1=cur_scan[0:PMAX, step_size:w],
                                   data2=cur_scan[0:PMAX, 0:w - step_size], op=nl.add)
                nisa.tensor_copy(dst=nxt_scan[0:PMAX, 0:step_size],
                                 src=cur_scan[0:PMAX, 0:step_size])
                cur_scan, nxt_scan = nxt_scan, cur_scan
                step_size *= 2

            nisa.tensor_tensor(dst=sub_excl[0:PMAX, 0:w], data1=cur_scan[0:PMAX, 0:w],
                               data2=sub_counts[0:PMAX, 0:w], op=nl.subtract)
            nisa.tensor_tensor(dst=sub_excl[0:PMAX, 0:w], data1=sub_excl[0:PMAX, 0:w],
                               data2=carry.ap(pattern=[[1, PMAX], [0, w]]), op=nl.add)
            nisa.tensor_tensor(dst=sub_excl[0:PMAX, 0:w], data1=sub_excl[0:PMAX, 0:w],
                               data2=chunk_base.ap(pattern=[[1, PMAX], [0, w]]), op=nl.add)
            nisa.dma_copy(dst=base.ap(pattern=[[chunk, PMAX], [1, w]],
                                      offset=ct * CHUNK_TILE),
                          src=sub_excl[0:PMAX, 0:w])

            nisa.tensor_tensor(dst=carry, data1=carry, data2=cur_scan[0:PMAX, w - 1:w],
                               op=nl.add)

        run_0 = nl.ndarray((1, S), dtype=nl.int32, buffer=nl.sbuf)
        run_1 = nl.ndarray((1, S), dtype=nl.int32, buffer=nl.sbuf)
        stage_0 = nl.ndarray((1, 2 * S), dtype=nl.int32, buffer=nl.sbuf)
        stage_1 = nl.ndarray((1, 2 * S), dtype=nl.int32, buffer=nl.sbuf)
        stage_2 = nl.ndarray((1, 2 * S), dtype=nl.int32, buffer=nl.sbuf)
        stage_3 = nl.ndarray((1, 2 * S), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=stage_0, value=-1)
        nisa.memset(dst=stage_1, value=-1)
        nisa.memset(dst=stage_2, value=-1)
        nisa.memset(dst=stage_3, value=-1)
        runs = (run_0, run_1)
        stages = (stage_0, stage_1, stage_2, stage_3)
        slot_counts_0 = nl.ndarray((1, ROWS), dtype=nl.int32, buffer=nl.sbuf)
        slot_counts_1 = nl.ndarray((1, ROWS), dtype=nl.int32, buffer=nl.sbuf)
        slot_offsets_0 = nl.ndarray((1, ROWS), dtype=nl.int32, buffer=nl.sbuf)
        slot_offsets_1 = nl.ndarray((1, ROWS), dtype=nl.int32, buffer=nl.sbuf)
        slot_counts = (slot_counts_0, slot_counts_1)
        slot_offsets = (slot_offsets_0, slot_offsets_1)
        step = 0

        for d in range(RADIX - 1, -1, -1):
            for tile_rev in range(n_tiles):
                tile_idx = n_tiles - 1 - tile_rev
                for load_row in range(ROWS):
                    load_part = load_row * GPSIMD_STRIDE
                    nisa.dma_copy(
                        dst=data[load_part:load_part + 1, 0:S],
                        src=work.ap(pattern=[[S, 1], [1, S]],
                                    offset=(tile_idx * ROWS + load_row) * S),
                    )
                nisa.tensor_scalar(dst=digit, data=data[0:PMAX, 0:S],
                                   op0=nl.right_shift, operand0=sh,
                                   op1=nl.bitwise_and, operand1=RADIX - 1)
                nisa.tensor_scalar(dst=digit, data=digit, op0=nl.equal, operand0=d)
                nisa.nonzero_with_count(dst=marks, src=digit,
                                        index_offset=0, padding_val=S)
                gathered = gathers[tile_rev % 2]
                nisa.nc_n_gather(dst=gathered, data=data,
                                 indices=marks[0:PMAX, 0:S].view(nl.uint32))
                cnt_row = slot_counts[tile_rev % 2]
                off_row = slot_offsets[tile_rev % 2]
                slot_base = d * NBLK + tile_idx * ROWS
                nisa.dma_copy(dst=cnt_row,
                              src=hist.ap(pattern=[[ROWS, 1], [1, ROWS]],
                                          offset=slot_base))
                nisa.dma_copy(dst=off_row,
                              src=base.ap(pattern=[[ROWS, 1], [1, ROWS]],
                                          offset=slot_base))
                for row_rev in range(ROWS):
                    row = ROWS - 1 - row_rev
                    part = row * GPSIMD_STRIDE
                    run = runs[step % 2]
                    cur = stages[step % 4]
                    prev = stages[(step - 1) % 4]
                    step += 1
                    nisa.dma_copy(dst=run, src=gathered[part:part + 1, 0:S])
                    nisa.tensor_copy(dst=cur[0:1, 0:S], src=run,
                                     engine=nisa.engine.vector)
                    nisa.tensor_copy(
                        dst=cur.ap(pattern=[[2 * S, 1], [1, S]], offset=0,
                                   scalar_offset=cnt_row[0:1, row:row + 1].view(nl.uint32),
                                   indirect_dim=1),
                        src=prev[0:1, 0:S],
                        engine=nisa.engine.vector,
                    )
                    nisa.dma_copy(
                        dst=out.ap(pattern=[[1, S]],
                                   scalar_offset=off_row[0:1, row:row + 1],
                                   indirect_dim=0),
                        src=cur[0:1, 0:S],
                    )
        return out


def _mark_step() -> None:
    from torch_xla.core import xla_model as xm
    xm.mark_step()


def _pick_block(n: int) -> int:
    block = BLOCK
    while block > MIN_BLOCK and PMAX * (block // 2) >= n:
        block //= 2
    return block


def _pass_args(keys: torch.Tensor, S: int) -> tuple:
    N = keys.numel()
    n_tiles = _div_ceil(N, PMAX * S)
    n_blocks = n_tiles * PMAX
    work_len = n_blocks * S + S
    tail = torch.full((work_len - N,), -1, dtype=torch.int32, device=keys.device)
    return torch.cat([keys.to(torch.int32), tail]).reshape(work_len, 1), n_blocks


_tuner = NkiAutotuner(radix_pass) if nki is not None else None
_last_autotune_config: dict = {}


def run(input: torch.Tensor, N: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    _kernel_assert(nki is not None, "Neuron SDK is not available")
    if N <= 1:
        return input.clone()

    keys = input.reshape(-1)
    _kernel_assert(keys.numel() == N, f"expected {N} elements, got {keys.numel()}")

    device, dtype = keys.device, keys.dtype
    S = _pick_block(N)
    if autotune:
        zero_shift = torch.full((PMAX, 1), 0, dtype=torch.int32, device=device)
        cfg = _tuner.tune_or_cached(
            shape_key=((N,), str(keys.dtype)),
            search_space=[SimpleNamespace(block_size=b) for b in (256, 512, 1024)
                          if MIN_BLOCK <= b <= BLOCK],
            args_fn=lambda cfg: (_pass_args(keys, cfg.block_size)[0], zero_shift,
                                 cfg.block_size, _pass_args(keys, cfg.block_size)[1]),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
        S = cfg.block_size
    work, n_blocks = _pass_args(keys, S)

    for shift in range(0, KEY_BITS, RADIX_BITS):
        shift_t = torch.full((PMAX, 1), shift, dtype=torch.int32, device=device)
        work = radix_pass(work, shift_t, S, n_blocks)
        _mark_step()

    return work.reshape(-1)[:N].to(dtype)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None

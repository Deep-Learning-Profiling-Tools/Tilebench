import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    PMAX = nl.tile_size.pmax          
    GPSIMD_STRIDE = 16                # nonzero_with_count touches partitions 0,16,...,112
    ROWS = PMAX // GPSIMD_STRIDE      # 8 blocks per scatter tile
except ImportError:
    nki = None

RADIX_BITS = 2
RADIX = 1 << RADIX_BITS
KEY_BITS = 32
MIN_BLOCK = 256

SBUF_PARTITION_BYTES = 192 * 1024   # trn2: 24 MB SBUF / 128 partitions
DGE_SCRATCH_BYTES = 16 * 1024       # backend's --dynamic-dma-scratch-size-per-partition
PHASE3_ROWS = 18

SAFETY = 2
_MAX_BLOCK = (SBUF_PARTITION_BYTES - DGE_SCRATCH_BYTES) // (PHASE3_ROWS * 4 * SAFETY)
BLOCK = 1 << (_MAX_BLOCK.bit_length() - 1)   # 1024 elements per block (one SBUF row)

CHUNK_TILE = 128


def _kernel_assert(cond, msg):
    if not cond:
        raise ValueError(f"[NCC_INKI016] Kernel validation exception: {msg}")


def _div_ceil(n, d):
    return (n + d - 1) // d


if nki is not None:

    @nki.jit
    def radix_pass(work, shift, S, NBLK):
        """One 2-bit radix pass: digit histogram -> global scan -> stable scatter.

        All three phases live in a single kernel. Splitting them across three
        ``@nki.jit`` calls lets XLA alias the intermediate HBM buffers against each
        other, which silently corrupts the scatter (verified on trn2: the scatter is
        exact when the histogram/offsets arrive as plain graph inputs and wrong when
        they are produced by sibling kernels in the same graph).

        Args:
            work:  ``(NBLK * S + S, 1)`` int32 keys in HBM; only ``[0, NBLK * S)`` is read.
            shift: ``(128, 1)`` int32 bit position of the current digit, replicated over
                   partitions so it can drive ``tensor_scalar``.
            S:     block size (elements per SBUF partition row).
            NBLK:  number of blocks; a multiple of 128.

        Returns:
            ``(NBLK * S + S, 1)`` int32; ``[0, NBLK * S)`` is the stably partitioned
            result and the trailing ``S`` elements absorb the last window's overhang.
        """
        M = RADIX * NBLK
        n_hist_tiles = NBLK // PMAX
        n_tiles = NBLK // ROWS

        out = nl.ndarray((NBLK * S + S, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        # Kernel-internal scratch: the only cheap way to turn the per-partition counts
        # into the partition-0 row the scan and the scatter both need.
        hist = nl.ndarray((M, 1), dtype=nl.int32, buffer=nl.hbm)

        sh = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sh, src=shift[0:PMAX, 0:1])

        # data keeps one extra, permanently zero column: nonzero_with_count pads its
        # index list with S, so gathered slots past the digit count read that sentinel.
        data = nl.ndarray((PMAX, S + 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=data, value=0)
        digit = nl.ndarray((PMAX, S), dtype=nl.int32, buffer=nl.sbuf)
        indicator = nl.ndarray((PMAX, S), dtype=nl.int32, buffer=nl.sbuf)
        # Kept exclusively for nonzero_with_count: sharing it with the histogram
        # indicator creates a compute/GpSimd write-after-write the scheduler does not
        # order, which silently corrupts the first tile's index list.
        marks = nl.ndarray((PMAX, S + 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=marks, value=0)
        # Double buffered: the eight per-row DMAs that drain a tile's gather race
        # against the next tile's nc_n_gather writing the same tile (cross-engine
        # write-after-read that the scheduler does not order).
        gathered_0 = nl.ndarray((PMAX, S), dtype=nl.int32, buffer=nl.sbuf)
        gathered_1 = nl.ndarray((PMAX, S), dtype=nl.int32, buffer=nl.sbuf)
        gathers = (gathered_0, gathered_1)
        tile_counts = nl.ndarray((PMAX, RADIX), dtype=nl.int32, buffer=nl.sbuf)

        # ---- phase 1: per-block digit histogram -----------------------------------
        # Phase 1 uses its own tiles. Reusing phase 3's buffers here leaves
        # cross-phase write-after-write pairs (compute vs GpSimd/DMA) that the
        # scheduler does not order, which corrupts the first scatter tile.
        hist_data = nl.ndarray((PMAX, S), dtype=nl.int32, buffer=nl.sbuf)
        hist_digit = nl.ndarray((PMAX, S), dtype=nl.int32, buffer=nl.sbuf)
        for tile_idx in nl.sequential_range(n_hist_tiles):
            nisa.dma_copy(dst=hist_data, src=work.ap(pattern=[[S, PMAX], [1, S]], offset=tile_idx * PMAX * S),)
            nisa.tensor_scalar(dst=hist_digit, data=hist_data, op0=nl.right_shift, operand0=sh, op1=nl.bitwise_and, operand1=RADIX - 1)
            for d in nl.static_range(RADIX):
                nisa.tensor_scalar(dst=indicator, data=hist_digit, op0=nl.equal, operand0=d)
                nisa.tensor_reduce(dst=tile_counts[0:PMAX, d:d + 1], data=indicator, op=nl.add, axis=(1,))
            for d in nl.static_range(RADIX):
                row = d * NBLK + tile_idx * PMAX
                # Written through .ap() (like the read below) so the dependency
                # tracker compares two flat views of the same buffer and orders them.
                nisa.dma_copy(dst=hist.ap(pattern=[[1, PMAX]], offset=row), src=tile_counts[0:PMAX, d:d + 1])

        # ---- phase 2: exclusive scan of the digit-major histogram ------------------
        # Three-level so *neither* the inner-scan state nor the outer-scan state depends
        # on M: the M counts are viewed as (128, C) with one contiguous chunk per
        # partition, but C itself is walked CHUNK_TILE columns at a time (a blocked scan,
        # the same "fixed buffer width, loop count grows with N" shape phase 1's
        # histogram loop already uses). A flat (1, M) scan would put 3*M int32 on
        # partition 0 alone, and even the once-two-level version above -- (PMAX, C) inner
        # buffers -- still put 4*C int32 on every partition; both blow the 192 KB
        # per-partition SBUF budget once N is large enough (measured: correct through
        # N=8e6, silently drops whole runs by N=2e7 -- same "accepts and spills" failure
        # phase 3's ``PHASE3_ROWS`` cap exists to avoid, just triggered from the M side
        # instead of the S side). Blocking the inner scan by CHUNK_TILE keeps every
        # phase-2 buffer's width fixed regardless of N.
        #
        # int32 Hillis/Steele throughout: tensor_tensor takes the integer ALU path and
        # stays exact, whereas tensor_tensor_scan accumulates in fp32 and drops low bits
        # once the running total passes 2**24 (N > ~16.7M).
        chunk = M // PMAX
        n_ctiles = _div_ceil(chunk, CHUNK_TILE)

        sub_counts = nl.ndarray((PMAX, CHUNK_TILE), dtype=nl.int32, buffer=nl.sbuf)
        sub_a = nl.ndarray((PMAX, CHUNK_TILE), dtype=nl.int32, buffer=nl.sbuf)
        sub_b = nl.ndarray((PMAX, CHUNK_TILE), dtype=nl.int32, buffer=nl.sbuf)
        sub_excl = nl.ndarray((PMAX, CHUNK_TILE), dtype=nl.int32, buffer=nl.sbuf)
        sub_sum = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        carry = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=carry, value=0)

        # Pass A: sum every sub-tile into `carry`, so it ends up holding each
        # partition's full chunk total (only the sum is needed here -- the per-element
        # prefix is recomputed from scratch in pass B once the outer base is known).
        for ct in nl.sequential_range(n_ctiles):
            w = min(CHUNK_TILE, chunk - ct * CHUNK_TILE)
            nisa.dma_copy(dst=sub_counts[0:PMAX, 0:w], src=hist.ap(pattern=[[chunk, PMAX], [1, w]], offset=ct * CHUNK_TILE))
            nisa.tensor_reduce(dst=sub_sum, data=sub_counts[0:PMAX, 0:w], op=nl.add, axis=(1,))
            nisa.tensor_tensor(dst=carry, data1=carry, data2=sub_sum, op=nl.add)

        # Outer level: exclusive scan of the 128 per-partition totals. The totals arrive
        # as a (128, 1) column and have to come back as one, so both transposes go
        # through HBM -- nc_transpose runs on the fp32 PE array and would round totals
        # > 2**24.
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
            nisa.tensor_tensor(dst=nxt_outer[0:1, step_size:PMAX], data1=cur_outer[0:1, step_size:PMAX], data2=cur_outer[0:1, 0:PMAX - step_size], op=nl.add)
            nisa.tensor_copy(dst=nxt_outer[0:1, 0:step_size], src=cur_outer[0:1, 0:step_size])
            cur_outer, nxt_outer = nxt_outer, cur_outer
            step_size *= 2
        nisa.tensor_tensor(dst=nxt_outer, data1=cur_outer, data2=totals, op=nl.subtract)
        # A second scratch buffer rather than reusing totals_hbm: writing and reading one
        # HBM buffer through two differently shaped .ap() views is exactly the pattern
        # the dependency tracker fails to order.
        base_hbm = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.hbm)
        nisa.dma_copy(dst=base_hbm.ap(pattern=[[PMAX, 1], [1, PMAX]]), src=nxt_outer)
        chunk_base = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=chunk_base, src=base_hbm.ap(pattern=[[1, PMAX]]))

        base = nl.ndarray((M, 1), dtype=nl.int32, buffer=nl.hbm)

        # Pass B: re-read each sub-tile, scan it locally, and offset by `carry` (the
        # running total of every earlier sub-tile in this partition) plus `chunk_base`
        # (the outer, cross-partition base) to get the true chunk-wide exclusive prefix.
        nisa.memset(dst=carry, value=0)
        for ct in nl.sequential_range(n_ctiles):
            w = min(CHUNK_TILE, chunk - ct * CHUNK_TILE)
            nisa.dma_copy(dst=sub_counts[0:PMAX, 0:w], src=hist.ap(pattern=[[chunk, PMAX], [1, w]], offset=ct * CHUNK_TILE))
            nisa.tensor_copy(dst=sub_a[0:PMAX, 0:w], src=sub_counts[0:PMAX, 0:w])
            cur_scan, nxt_scan = sub_a, sub_b
            step_size = 1
            while step_size < w:
                nisa.tensor_tensor(dst=nxt_scan[0:PMAX, step_size:w], data1=cur_scan[0:PMAX, step_size:w], data2=cur_scan[0:PMAX, 0:w - step_size], op=nl.add)
                nisa.tensor_copy(dst=nxt_scan[0:PMAX, 0:step_size], src=cur_scan[0:PMAX, 0:step_size])
                cur_scan, nxt_scan = nxt_scan, cur_scan
                step_size *= 2

            # exclusive[p][i] = inclusive[p][i] - counts[p][i] + carry[p] + chunk_base[p]
            nisa.tensor_tensor(dst=sub_excl[0:PMAX, 0:w], data1=cur_scan[0:PMAX, 0:w], data2=sub_counts[0:PMAX, 0:w], op=nl.subtract)
            # tensor_scalar rejects an int32 vector operand, so the per-partition base
            # is broadcast across the sub-tile with a zero-stride free dimension instead.
            nisa.tensor_tensor(dst=sub_excl[0:PMAX, 0:w], data1=sub_excl[0:PMAX, 0:w], data2=carry.ap(pattern=[[1, PMAX], [0, w]]), op=nl.add)
            nisa.tensor_tensor(dst=sub_excl[0:PMAX, 0:w], data1=sub_excl[0:PMAX, 0:w], data2=chunk_base.ap(pattern=[[1, PMAX], [0, w]]), op=nl.add)
            nisa.dma_copy(dst=base.ap(pattern=[[chunk, PMAX], [1, w]], offset=ct * CHUNK_TILE), src=sub_excl[0:PMAX, 0:w])

            nisa.tensor_tensor(dst=carry, data1=carry, data2=cur_scan[0:PMAX, w - 1:w], op=nl.add)

        # ---- phase 3: stable partition + scatter -----------------------------------
        # The output stream is O = concat_{d, b} run(d, b) and offsets[d][b] is where
        # run(d, b) starts in it. Blocks are visited in *decreasing* (d, b) order while
        # a staging row holds O[offsets[d][b] : +S]; the invariant is restored by
        # writing run(d, b) at 0 and re-placing the previous window at counts[d][b].
        # Every S-element write is then correct for each position it covers, so the
        # writes are idempotent -- which matters because NKI gives no ordering
        # guarantee between independent indirect DMA writes.
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
        # Only the ROWS counts/offsets of the current tile are held in SBUF, so phase 3
        # is also independent of M. Double buffered for the same cross-engine reason as
        # the gather output.
        slot_counts_0 = nl.ndarray((1, ROWS), dtype=nl.int32, buffer=nl.sbuf)
        slot_counts_1 = nl.ndarray((1, ROWS), dtype=nl.int32, buffer=nl.sbuf)
        slot_offsets_0 = nl.ndarray((1, ROWS), dtype=nl.int32, buffer=nl.sbuf)
        slot_offsets_1 = nl.ndarray((1, ROWS), dtype=nl.int32, buffer=nl.sbuf)
        slot_counts = (slot_counts_0, slot_counts_1)
        slot_offsets = (slot_offsets_0, slot_offsets_1)
        step = 0

        for d in nl.static_range(RADIX - 1, -1, -1):
            for tile_rev in nl.sequential_range(n_tiles):
                tile_idx = n_tiles - 1 - tile_rev
                # One DMA per row rather than a single partition-strided DMA: the
                # dependency tracker does not see that a `tile[0:128:16]` write
                # overlaps the following full-tile read, and skips the semaphore.
                for load_row in nl.static_range(ROWS):
                    load_part = load_row * GPSIMD_STRIDE
                    nisa.dma_copy(
                        dst=data[load_part:load_part + 1, 0:S],
                        src=work.ap(pattern=[[S, 1], [1, S]],
                        offset=(tile_idx * ROWS + load_row) * S),
                    )
                nisa.tensor_scalar(dst=digit, data=data[0:PMAX, 0:S], op0=nl.right_shift, operand0=sh, op1=nl.bitwise_and, operand1=RADIX - 1)
                nisa.tensor_scalar(dst=digit, data=digit, op0=nl.equal, operand0=d)
                nisa.nonzero_with_count(dst=marks, src=digit, index_offset=0, padding_val=S)
                gathered = gathers[tile_rev % 2]
                nisa.nc_n_gather(dst=gathered, data=data, indices=marks[0:PMAX, 0:S].view(nl.uint32))
                cnt_row = slot_counts[tile_rev % 2]
                off_row = slot_offsets[tile_rev % 2]
                slot_base = d * NBLK + tile_idx * ROWS
                nisa.dma_copy(dst=cnt_row, src=hist.ap(pattern=[[ROWS, 1], [1, ROWS]], offset=slot_base))
                nisa.dma_copy(dst=off_row, src=base.ap(pattern=[[ROWS, 1], [1, ROWS]], offset=slot_base))
                for row_rev in nl.static_range(ROWS):
                    row = ROWS - 1 - row_rev
                    part = row * GPSIMD_STRIDE
                    run = runs[step % 2]
                    cur = stages[step % 4]
                    prev = stages[(step - 1) % 4]
                    step += 1
                    nisa.dma_copy(dst=run, src=gathered[part:part + 1, 0:S])
                    # Both copies are pinned to one compute engine, whose instruction
                    # stream is in order, so the second reliably repairs the first's tail.
                    nisa.tensor_copy(dst=cur[0:1, 0:S], src=run, engine=nisa.engine.vector)
                    nisa.tensor_copy(
                        dst=cur.ap(pattern=[[2 * S, 1], [1, S]], offset=0,
                                   scalar_offset=cnt_row[0:1, row:row + 1].view(nl.uint32),
                                   indirect_dim=1),
                        src=prev[0:1, 0:S],
                        engine=nisa.engine.vector,
                    )
                    nisa.dma_copy(
                        dst=out.ap(pattern=[[1, S]], scalar_offset=off_row[0:1, row:row + 1], indirect_dim=0),
                        src=cur[0:1, 0:S],
                    )
        return out


def _mark_step() -> None:
    """Close the current XLA graph (deferred import: GPU-only hosts lack torch_xla)."""
    from torch_xla.core import xla_model as xm
    xm.mark_step()


def _pick_block(n: int) -> int:
    """Smallest power-of-two block size that still fills all 128 partitions."""
    block = BLOCK
    while block > MIN_BLOCK and PMAX * (block // 2) >= n:
        block //= 2
    return block


def run(input: torch.Tensor, N: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    
    _kernel_assert(nki is not None, "Neuron SDK is not available")
    if N <= 1:
        return input.clone()

    keys = input.reshape(-1)
    _kernel_assert(keys.numel() == N, f"expected {N} elements, got {keys.numel()}")

    device, dtype = keys.device, keys.dtype
    S = _pick_block(N)
    n_tiles = _div_ceil(N, PMAX * S)
    n_blocks = n_tiles * PMAX
    n_pad = n_blocks * S
    work_len = n_pad + S            # + overhang guard for the fixed-length window writes

    # 0xFFFFFFFF is the largest unsigned key, so padding always sorts past the real data.
    tail = torch.full((work_len - N,), -1, dtype=torch.int32, device=device)
    work = torch.cat([keys.to(torch.int32), tail]).reshape(work_len, 1)

    for shift in range(0, KEY_BITS, RADIX_BITS):
        shift_t = torch.full((PMAX, 1), shift, dtype=torch.int32, device=device)
        work = radix_pass(work, shift_t, S, n_blocks)
        _mark_step()

    return work.reshape(-1)[:N].to(dtype)


def get_last_config() -> dict | None:
    return None

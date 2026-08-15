"""NKI 5-point Jacobi stencil matching ``impl_torch.py``.

Reference semantics (see ``impl_torch.py``)::

    out = input.clone()
    out[1:rows-1, 1:cols-1] = 0.25 * (input[0:rows-2, 1:cols-1]
                                      + input[2:rows,   1:cols-1]
                                      + input[1:rows-1, 0:cols-2]
                                      + input[1:rows-1, 2:cols])

i.e. every interior cell becomes the average of its four neighbours while the
four borders (row 0, row ``rows-1``, column 0, column ``cols-1``) pass through
unchanged (Dirichlet boundary).

Algorithm
---------
``run()`` hands the kernel a zero-padded ``(rows+2, cols+2)`` grid, so padded
row/column ``k+1`` holds original row/column ``k``.  The interior rows
``1 .. rows-2`` are walked in ``(PMAX x col_block)`` blocks:

* **Rows** are the partition axis, so the up/centre/down neighbours cannot be
  reached by shifting inside a tile -- they are three separate DMAs of the same
  ``(p_sz, cb+2)`` window at padded row offsets ``0/1/2``.  ``p_sz`` is clamped
  to ``min(PMAX, ...)``, which is how the ragged last row block is handled (the
  old ``nl.arange`` + ``mask=`` version is not expressible in the current API,
  and a clamped tile is cheaper anyway).
* **Columns** are the free axis, so left/right are just Python slices of the
  already-loaded centre tile (``ctr[:, 0:cb]`` / ``ctr[:, 2:cb+2]``) -- no extra
  traffic.  Columns are additionally blocked so the per-partition SBUF footprint
  stays bounded for the largest sweep point (``cols = 10240`` in fp32 does not
  fit three full-width windows plus an fp32 accumulator).
* The additions are accumulated in fp32 in the same left-to-right order as the
  torch reference, so fp16/bf16 inputs do not inherit the reference's
  intermediate rounding.

Boundary handling is *static*: ``rows``/``cols`` are plain Python ints traced
into the kernel, so which cells are on a border is a compile-time fact and
ordinary Python ``if`` is enough.  The old version built device-side boolean
masks (``nl.arange`` comparisons + ``nl.broadcast_to`` + ``nl.where``) for a
condition that is constant per block; none of that machinery exists in the
current API and none of it is needed.

* **Column borders** are overwritten with the centre value after the average has
  been computed, with a one-column ``nisa.tensor_copy`` in the blocks that own
  column ``0`` / ``cols-1``.
* **Row borders** are *peeled* rather than overwritten: global rows ``0`` and
  ``rows-1`` are copied straight through and the blocked loop only covers rows
  ``1 .. rows-2``.  Overwriting them in place would need a one-partition access
  at an arbitrary partition (e.g. row ``rows-1`` landing on partition 127 of a
  full block), which the hardware rejects -- ``BIR verification failed ...
  Invalid access of 1 partitions starting at partition 127``.  Peeling keeps
  every access anchored at partition 0.
"""

import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax          # 128 partitions
except ImportError:
    nki = None
    PMAX = 128

# Per-partition SBUF spent on the three row windows + accumulator + output tile
# (trn2 has 192KB per partition).
SBUF_BUDGET_BYTES = 96 * 1024


def div_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def kernel_assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"[jacobi_stencil_2d NKI] {message}")


def _choose_col_block(itemsize: int, cols: int) -> int:
    """Largest column block whose SBUF working set fits the per-partition budget.

    Per output column a block costs three window elements (input dtype), one
    fp32 accumulator element and one output element.
    """
    per_col = 3 * itemsize + 4 + itemsize
    col_block = cols
    while col_block > 128 and per_col * (col_block + 2) > SBUF_BUDGET_BYTES:
        col_block = div_ceil(col_block, 2)
    return col_block


if nki is not None:

    @nki.jit
    def jacobi_kernel(padded_input, rows, cols, col_block):
        """5-point Jacobi stencil with pass-through borders.

        Args:
            padded_input: (rows + 2, cols + 2) zero-padded grid
            rows, cols: original grid extents (compile-time constants)
            col_block: columns computed per block (compile-time constant)

        Returns:
            (rows, cols)
        """
        dtype = padded_input.dtype

        # NKI kernels cannot `raise`; run() does the friendly validation, these
        # are the in-kernel invariants.
        assert padded_input.shape[0] == rows + 2
        assert padded_input.shape[1] == cols + 2

        output_hbm = nl.ndarray((rows, cols), dtype=dtype, buffer=nl.shared_hbm)

        # --- border rows: straight pass-through --------------------------
        border_rows = [0] if rows == 1 else [0, rows - 1]
        for r in border_rows:
            row_tile = nl.ndarray((1, cols), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=row_tile,
                          src=padded_input[r + 1:r + 2, 1:cols + 1])
            nisa.dma_copy(dst=output_hbm[r:r + 1, 0:cols], src=row_tile)

        # --- interior rows 1 .. rows-2 ------------------------------------
        n_interior = max(0, rows - 2)
        n_row_blocks = div_ceil(n_interior, PMAX)
        n_col_blocks = div_ceil(cols, col_block)

        for i in range(n_row_blocks):
            row0 = 1 + i * PMAX
            p_sz = min(PMAX, rows - 1 - row0)

            for j in range(n_col_blocks):
                c0 = j * col_block
                cb = min(col_block, cols - c0)

                # Window k holds padded rows [row0+k, row0+k+p_sz): k=0 is the
                # "up" neighbour of global rows [row0, row0+p_sz), k=1 the rows
                # themselves, k=2 the "down" neighbour.
                up = nl.ndarray((p_sz, cb + 2), dtype=dtype, buffer=nl.sbuf)
                ctr = nl.ndarray((p_sz, cb + 2), dtype=dtype, buffer=nl.sbuf)
                dn = nl.ndarray((p_sz, cb + 2), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=up,
                    src=padded_input[row0:row0 + p_sz, c0:c0 + cb + 2],
                )
                nisa.dma_copy(
                    dst=ctr,
                    src=padded_input[row0 + 1:row0 + 1 + p_sz, c0:c0 + cb + 2],
                )
                nisa.dma_copy(
                    dst=dn,
                    src=padded_input[row0 + 2:row0 + 2 + p_sz, c0:c0 + cb + 2],
                )

                # up + down + left + right, in the reference's evaluation order.
                acc = nl.ndarray((p_sz, cb), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=acc, data1=up[0:p_sz, 1:cb + 1],
                                   data2=dn[0:p_sz, 1:cb + 1], op=nl.add)
                nisa.tensor_tensor(dst=acc, data1=acc, data2=ctr[0:p_sz, 0:cb],
                                   op=nl.add)
                nisa.tensor_tensor(dst=acc, data1=acc, data2=ctr[0:p_sz, 2:cb + 2],
                                   op=nl.add)

                result = nl.ndarray((p_sz, cb), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=result, data=acc, op0=nl.multiply,
                                   operand0=0.25)

                # Column borders pass through unchanged.  Padded column j+1
                # holds original column j, so the centre of local column 0 is
                # ctr[:, 1] and the centre of local column cb-1 is ctr[:, cb].
                if c0 == 0:
                    nisa.tensor_copy(dst=result[0:p_sz, 0:1],
                                     src=ctr[0:p_sz, 1:2])
                if c0 + cb == cols:
                    nisa.tensor_copy(dst=result[0:p_sz, cb - 1:cb],
                                     src=ctr[0:p_sz, cb:cb + 1])

                nisa.dma_copy(dst=output_hbm[row0:row0 + p_sz, c0:c0 + cb],
                              src=result)

        return output_hbm


def run(input: torch.Tensor, rows: int, cols: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    kernel_assert(rows >= 1 and cols >= 1, "rows and cols must be >= 1")
    kernel_assert(input.shape[0] == rows and input.shape[1] == cols,
                  "input shape must be (rows, cols)")

    padded = torch.nn.functional.pad(input, (1, 1, 1, 1))
    col_block = _choose_col_block(input.element_size(), cols)
    return jacobi_kernel(padded, rows, cols, col_block)


def get_last_config() -> dict | None:
    return None

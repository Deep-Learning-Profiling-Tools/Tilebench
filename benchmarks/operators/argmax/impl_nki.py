import torch

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

# ``nisa.max8`` / ``nisa.nc_find_index8`` require between 8 and 16384 elements
# per partition, so every free-dimension chunk must stay inside that window.
MAX8_MIN_ELEMS = 8
MAX8_MAX_ELEMS = 16384


def _free_chunks(free_dim: int, max_chunk: int = MAX8_MAX_ELEMS):
    """Split ``free_dim`` into contiguous, near-equal chunks of at most ``max_chunk``.

    An even split (rather than ``[max_chunk, max_chunk, ..., remainder]``) keeps the
    last chunk from becoming a tiny tail: for ``free_dim >= 8`` every produced chunk
    is guaranteed to hold at least 8 elements, which is the ``max8`` lower bound.

    Returns a list of ``(start, end)`` Python tuples evaluated at trace time.
    """
    num_chunks = (free_dim + max_chunk - 1) // max_chunk
    base = free_dim // num_chunks
    rem = free_dim % num_chunks

    chunks = []
    start = 0
    for c in range(num_chunks):
        size = base + (1 if c < rem else 0)
        chunks.append((start, start + size))
        start += size
    return chunks


@nki.jit
def argmax_kernel(a_input):
    """Row-wise argmax of a 2D tensor: ``out[p] = argmax(a_input[p, :])``.

    Matches ``torch.argmax(x, dim=1)`` semantics, including tie-breaking on the
    *first* occurrence of the maximum value:

    * within a free-dimension chunk, ``nisa.nc_find_index8`` returns the index of
      the first occurrence of each of the top-8 values;
    * across chunks, a new chunk only replaces the running best when its maximum is
      *strictly* greater (``nl.greater``), so the earliest chunk wins on a tie.

    The running (max, index) pair is combined branch-free in fp32 as
    ``idx = idx + better * (chunk_idx - idx)`` which avoids needing a select over
    integer index tiles.

    Args:
        a_input: 2D HBM tensor of shape [P, F], float dtype.

    Returns:
        HBM tensor of shape [P, 1], int32, holding the argmax column index per row.
    """
    P, F = a_input.shape

    num_par_blocks = (P + PMAX - 1) // PMAX
    chunks = _free_chunks(F)

    hbm_result_tile = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.shared_hbm)

    for i in range(num_par_blocks):
        p_start = i * PMAX
        p_end = min(p_start + PMAX, P)
        p_sz = p_end - p_start

        for c in range(len(chunks)):
            f_start = chunks[c][0]
            f_end = chunks[c][1]
            f_sz = f_end - f_start
            # max8/nc_find_index8 need >= 8 elements per partition. For a very
            # narrow input the tile is padded with -inf; because the padding sits
            # *after* the real data and is the smallest possible value, the
            # first-occurrence index reported by nc_find_index8 always lands on a
            # real element (index < f_sz).
            tile_sz = max(f_sz, MAX8_MIN_ELEMS)

            a_tile = nl.ndarray((p_sz, tile_sz), dtype=a_input.dtype, buffer=nl.sbuf)
            if tile_sz != f_sz:
                nisa.memset(dst=a_tile, value=float('-inf'))
            nisa.dma_copy(dst=a_tile[0:p_sz, 0:f_sz], src=a_input[p_start:p_end, f_start:f_end])

            # Top-8 values (descending) and the index of their first occurrence.
            # ``vals`` must share the dtype of ``data`` so the value match is exact.
            top8 = nl.ndarray((p_sz, 8), dtype=a_input.dtype, buffer=nl.sbuf)
            nisa.max8(dst=top8, src=a_tile)

            idx8 = nl.ndarray((p_sz, 8), dtype=nl.uint16, buffer=nl.sbuf)
            nisa.nc_find_index8(dst=idx8, data=a_tile, vals=top8)

            chunk_max = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=chunk_max, src=top8[0:p_sz, 0:1])

            chunk_idx = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            if f_start == 0:
                nisa.tensor_copy(dst=chunk_idx, src=idx8[0:p_sz, 0:1])
            else:
                nisa.tensor_scalar(
                    dst=chunk_idx,
                    data=idx8[0:p_sz, 0:1],
                    op0=nl.add,
                    operand0=float(f_start),
                )

            if c == 0:
                cur_max = chunk_max
                cur_idx = chunk_idx
            else:
                # better = 1.0 where this chunk *strictly* beats the running max,
                # so an earlier chunk keeps the index on a tie (first occurrence).
                better = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=better, data1=chunk_max, data2=cur_max, op=nl.greater)

                new_max = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=new_max, data1=cur_max, data2=chunk_max, op=nl.maximum)

                # new_idx = cur_idx + better * (chunk_idx - cur_idx)
                delta = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=delta, data1=chunk_idx, data2=cur_idx, op=nl.subtract)

                masked_delta = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=masked_delta, data1=delta, data2=better, op=nl.multiply)

                new_idx = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=new_idx, data1=cur_idx, data2=masked_delta, op=nl.add)

                cur_max = new_max
                cur_idx = new_idx

        out_tile = nl.ndarray((p_sz, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_tile, src=cur_idx)
        nisa.dma_copy(dst=hbm_result_tile[p_start:p_end, 0:1], src=out_tile)

    return hbm_result_tile


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("argmax NKI: int8 not supported")
    if x.dim() != 2:
        raise NotImplementedError("argmax NKI: expects a 2D input (rows, cols)")
    if dim != 1:
        raise NotImplementedError("argmax NKI: only dim=1 (row-wise) is supported")

    result = argmax_kernel(x)
    return result.reshape(-1).to(torch.int64)

def get_last_config() -> dict | None:
    return None

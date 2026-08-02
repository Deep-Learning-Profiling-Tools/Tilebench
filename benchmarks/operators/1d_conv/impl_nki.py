import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def conv1d_kernel(input_1d, kernel_1d, output_size, kernel_size, F):
        # output[o] = sum_k kernel[k] * input[o + k]. Output positions are
        # assigned to (partition, free) as o = p*F + f -- each of the PMAX
        # partitions owns one contiguous chunk of F output positions, tiled
        # along the free dim in chunk_size pieces (vector_add's layout).
        # This replaces an earlier one-output-row-per-partition design that
        # needed ceil(output_size/PMAX) partition-blocks -- ~7800 for the
        # smallest configured case (output_size ~= 1e6) -- which blew the
        # compiler's 5M-instruction budget (NCC_EBVF030) well before reaching
        # the *largest* configured case. Folding most of the size into the
        # free dimension instead of the block count keeps the instruction
        # count in the thousands regardless of output_size.
        chunk_size = 8192
        num_chunks = (F + chunk_size - 1) // chunk_size

        hbm_result = nl.ndarray((output_size, 1), dtype=input_1d.dtype, buffer=nl.hbm)
        partition_index = nl.arange(PMAX)[:, None]
        zero_index = nl.arange(1)[None, :]

        for c in range(num_chunks):
            free_offset = c * chunk_size
            free_index = nl.arange(chunk_size)[None, :]
            global_idx = partition_index * F + free_offset + free_index
            mask = global_idx < output_size

            acc = nl.zeros((PMAX, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
            for k in range(kernel_size):
                x_tile = nl.load(input_1d[partition_index * F + free_offset + free_index + k, zero_index],
                                  mask=mask, dtype=nl.float32)
                w_scalar = nl.load(kernel_1d[k + nl.arange(1)[:, None], zero_index], dtype=nl.float32)
                w_broadcast = nl.broadcast_to(w_scalar, shape=(PMAX, chunk_size))
                acc[...] = nl.add(acc, nl.multiply(x_tile, w_broadcast))

            result_tile = nl.add(acc, 0.0, dtype=input_1d.dtype)
            nl.store(hbm_result[partition_index * F + free_offset + free_index, zero_index],
                     value=result_tile, mask=mask)

        return hbm_result


def run(input: torch.Tensor, kernel: torch.Tensor, input_size: int, kernel_size: int,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    output_size = input_size - kernel_size + 1
    F = (output_size + PMAX - 1) // PMAX
    input_2d = input.reshape(-1, 1)
    kernel_2d = kernel.reshape(-1, 1)

    # input_1d[p*F + f + k] is read for p up to PMAX-1, f up to F-1 (padded
    # chunk tail included), k up to kernel_size-1 -- pad so that read always
    # lands inside the allocation regardless of the tail mask.
    padded_size = PMAX * F + kernel_size - 1
    if padded_size > input_size:
        input_2d = torch.nn.functional.pad(input_2d, (0, 0, 0, padded_size - input_size))

    result = conv1d_kernel(input_2d, kernel_2d, output_size, kernel_size, F)
    return result.reshape(-1)


def get_last_config() -> dict | None:
    return None

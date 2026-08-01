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
    def conv1d_kernel(input_1d, kernel_1d, output_size, kernel_size):
        # output[o] = sum_k kernel[k] * input[o + k]. output_size positions
        # sit on the partition dim (PMAX per block); each of the kernel_size
        # taps is one shifted (PMAX, 1) load + scalar-broadcast multiply-add.
        num_blocks = (output_size + (PMAX - 1)) // PMAX

        hbm_result = nl.ndarray((output_size, 1), dtype=input_1d.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            zero_index = nl.arange(1)[None, :]
            mask_p = partition_index < (output_size - offset)

            acc = nl.zeros((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for k in range(kernel_size):
                x_tile = nl.load(input_1d[offset + partition_index + k, zero_index],
                                  dtype=nl.float32)
                w_scalar = nl.load(kernel_1d[k + nl.arange(1)[:, None], zero_index], dtype=nl.float32)
                w_broadcast = nl.broadcast_to(w_scalar, shape=(PMAX, 1))
                acc[...] = nl.add(acc, nl.multiply(x_tile, w_broadcast))

            result_tile = nl.add(acc, 0.0, dtype=input_1d.dtype)
            nl.store(hbm_result[offset + partition_index, zero_index], value=result_tile, mask=mask_p)

        return hbm_result


def run(input: torch.Tensor, kernel: torch.Tensor, input_size: int, kernel_size: int,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    output_size = input_size - kernel_size + 1
    input_2d = input.reshape(-1, 1)
    kernel_2d = kernel.reshape(-1, 1)

    # input_1d[offset + p + k] is read for k up to kernel_size-1 and p up to
    # PMAX-1 in the last (possibly partial) block -- pad so that read always
    # lands inside the allocation regardless of the tail block's mask.
    padded_size = ((output_size + PMAX - 1) // PMAX) * PMAX + kernel_size - 1
    if padded_size > input_size:
        input_2d = torch.nn.functional.pad(input_2d, (0, 0, 0, padded_size - input_size))

    result = conv1d_kernel(input_2d, kernel_2d, output_size, kernel_size)
    return result.reshape(-1)


def get_last_config() -> dict | None:
    return None

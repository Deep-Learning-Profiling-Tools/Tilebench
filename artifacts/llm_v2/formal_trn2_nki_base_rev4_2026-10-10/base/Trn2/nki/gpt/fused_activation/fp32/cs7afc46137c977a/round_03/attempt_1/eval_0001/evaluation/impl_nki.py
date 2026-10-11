import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _fused_activation(x, gate, bias, free_size, compute_free_size):
    out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)

    partitions = 128
    core_elements = x.shape[0] // nl.num_programs()
    core_offset = nl.program_id(0) * core_elements
    tile_elements = partitions * free_size
    tile_count = (core_elements + tile_elements - 1) // tile_elements

    for i in range(tile_count):
        local_offset = i * tile_elements
        offset = core_offset + local_offset
        width = min(free_size, (core_elements - local_offset) // partitions)

        xt = nl.ndarray(
            (partitions, width), dtype=nl.float32, buffer=nl.sbuf
        )
        gt = nl.ndarray(
            (partitions, width), dtype=nl.float32, buffer=nl.sbuf
        )
        bt = nl.ndarray(
            (partitions, width), dtype=nl.float32, buffer=nl.sbuf
        )

        nisa.dma_copy(
            dst=xt,
            src=x.ap(
                pattern=[[width, partitions], [1, width]],
                offset=offset,
            ),
        )
        nisa.dma_copy(
            dst=gt,
            src=gate.ap(
                pattern=[[width, partitions], [1, width]],
                offset=offset,
            ),
        )
        nisa.dma_copy(
            dst=bt,
            src=bias.ap(
                pattern=[[width, partitions], [1, width]],
                offset=offset,
            ),
        )

        # Large DMA transfers, with smaller independent compute regions.
        # Vector work on later regions can overlap ScalarE activation.
        for j in range((width + compute_free_size - 1) // compute_free_size):
            begin = j * compute_free_size
            end = min(width, begin + compute_free_size)

            nisa.tensor_tensor(
                dst=xt[:, begin:end],
                data1=xt[:, begin:end],
                data2=gt[:, begin:end],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=xt[:, begin:end],
                data1=xt[:, begin:end],
                data2=bt[:, begin:end],
                op=nl.add,
            )
            nisa.activation(
                dst=gt[:, begin:end],
                op=nl.silu,
                data=xt[:, begin:end],
            )

        nisa.dma_copy(
            dst=out.ap(
                pattern=[[width, partitions], [1, width]],
                offset=offset,
            ),
            src=gt,
        )

    return out


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, **kwargs):
    global _last_config

    if len(x.shape) != 1 or gate.shape != x.shape or bias.shape != x.shape:
        raise ValueError("Inputs must be identically shaped vectors.")
    if (
        x.dtype != torch.float32
        or gate.dtype != torch.float32
        or bias.dtype != torch.float32
    ):
        raise ValueError("Inputs must have dtype torch.float32.")
    if x.device != gate.device or x.device != bias.device:
        raise ValueError("Inputs must be on the same device.")

    n = x.shape[0]
    cores = 2
    partitions = 128
    if n == 0 or n % (cores * partitions) != 0:
        raise ValueError("Input length must be a positive multiple of 256.")

    free_size = min(8192, max(1, n // (cores * partitions * 4)))
    compute_free_size = min(1024, free_size)
    core_elements = n // cores
    tile_elements = partitions * free_size
    remainder = core_elements % tile_elements

    _last_config = {
        "lnc": cores,
        "partitions": partitions,
        "free_size": free_size,
        "compute_free_size": compute_free_size,
        "tile_elements": tile_elements,
        "tiles_per_core": (core_elements + tile_elements - 1) // tile_elements,
        "tail_free_size": remainder // partitions,
        "activation": "native_silu",
        "dtype": "fp32",
        "loop": "unrolled_bulk_dma_microcompute",
    }

    return _fused_activation[2](
        x, gate, bias, free_size, compute_free_size
    )


def get_last_config() -> dict:
    return dict(_last_config)

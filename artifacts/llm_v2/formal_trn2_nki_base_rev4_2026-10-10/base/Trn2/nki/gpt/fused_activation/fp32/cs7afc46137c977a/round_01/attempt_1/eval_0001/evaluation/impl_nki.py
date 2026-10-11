import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _fused_activation(x, gate, bias, free_size):
    out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)

    partitions = 128
    tile_elements = partitions * free_size
    tile_count = x.shape[0] // tile_elements
    tiles_per_core = tile_count // nl.num_programs()
    first_tile = nl.program_id(0) * tiles_per_core

    xv = x.reshape((tile_count, partitions, free_size))
    gv = gate.reshape((tile_count, partitions, free_size))
    bv = bias.reshape((tile_count, partitions, free_size))
    ov = out.reshape((tile_count, partitions, free_size))

    for i in range(tiles_per_core):
        tile = first_tile + i
        xt = nl.ndarray(
            (partitions, free_size), dtype=nl.float32, buffer=nl.sbuf
        )
        gt = nl.ndarray(
            (partitions, free_size), dtype=nl.float32, buffer=nl.sbuf
        )
        bt = nl.ndarray(
            (partitions, free_size), dtype=nl.float32, buffer=nl.sbuf
        )

        nisa.dma_copy(dst=xt, src=xv[tile])
        nisa.dma_copy(dst=gt, src=gv[tile])
        nisa.dma_copy(dst=bt, src=bv[tile])

        nisa.tensor_tensor(dst=xt, data1=xt, data2=gt, op=nl.multiply)
        nisa.tensor_tensor(dst=xt, data1=xt, data2=bt, op=nl.add)
        nisa.activation(dst=gt, op=nl.silu, data=xt)

        nisa.dma_copy(dst=ov[tile], src=gt)

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
    partitions = 128
    cores = 2

    # Keep at least four streaming tiles per core, capped at 4096
    # elements per partition. Every configured size divides exactly.
    free_size = min(4096, max(1, n // (cores * partitions * 4)))
    tile_elements = partitions * free_size
    if n % (cores * tile_elements) != 0:
        raise ValueError("Input length must divide the selected core tiling.")

    _last_config = {
        "lnc": cores,
        "partitions": partitions,
        "free_size": free_size,
        "tile_elements": tile_elements,
        "tiles_per_core": n // (cores * tile_elements),
        "activation": "native_silu",
        "dtype": "fp32",
        "loop": "unrolled",
    }

    return _fused_activation[2](x, gate, bias, free_size)


def get_last_config() -> dict:
    return dict(_last_config)

import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _fused_activation(x, gate, bias, free_size, stages):
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

    # A bounded SBUF ring keeps upcoming DMA loads independent of the
    # current tile's arithmetic and output transfer.
    xt = nl.ndarray(
        (partitions, stages, free_size), dtype=nl.float32, buffer=nl.sbuf
    )
    gt = nl.ndarray(
        (partitions, stages, free_size), dtype=nl.float32, buffer=nl.sbuf
    )
    bt = nl.ndarray(
        (partitions, stages, free_size), dtype=nl.float32, buffer=nl.sbuf
    )

    for j in range(stages - 1):
        nisa.dma_copy(dst=xt[:, j, :], src=xv[first_tile + j])
        nisa.dma_copy(dst=gt[:, j, :], src=gv[first_tile + j])
        nisa.dma_copy(dst=bt[:, j, :], src=bv[first_tile + j])

    for i in range(tiles_per_core):
        ahead = i + stages - 1
        if ahead < tiles_per_core:
            load_slot = ahead % stages
            tile = first_tile + ahead
            nisa.dma_copy(dst=xt[:, load_slot, :], src=xv[tile])
            nisa.dma_copy(dst=gt[:, load_slot, :], src=gv[tile])
            nisa.dma_copy(dst=bt[:, load_slot, :], src=bv[tile])

        slot = i % stages
        nisa.tensor_tensor(
            dst=xt[:, slot, :],
            data1=xt[:, slot, :],
            data2=gt[:, slot, :],
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=xt[:, slot, :],
            data1=xt[:, slot, :],
            data2=bt[:, slot, :],
            op=nl.add,
        )
        nisa.activation(
            dst=gt[:, slot, :],
            op=nl.silu,
            data=xt[:, slot, :],
        )
        nisa.dma_copy(dst=ov[first_tile + i], src=gt[:, slot, :])

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
    free_size = min(4096, max(1, n // (cores * partitions * 4)))
    tile_elements = partitions * free_size

    if n % (cores * tile_elements) != 0:
        raise ValueError("Input length must divide the selected core tiling.")

    tiles_per_core = n // (cores * tile_elements)
    stages = min(3, tiles_per_core)

    _last_config = {
        "lnc": cores,
        "partitions": partitions,
        "free_size": free_size,
        "tile_elements": tile_elements,
        "tiles_per_core": tiles_per_core,
        "stages": stages,
        "prefetch_distance": stages - 1,
        "activation": "native_silu",
        "dtype": "fp32",
        "loop": "unrolled_sbuf_ring",
    }

    return _fused_activation[2](x, gate, bias, free_size, stages)


def get_last_config() -> dict:
    return dict(_last_config)

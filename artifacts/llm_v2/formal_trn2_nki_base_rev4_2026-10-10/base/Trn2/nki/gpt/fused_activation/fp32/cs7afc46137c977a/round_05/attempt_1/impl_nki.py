import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit(compile_config=nki.compiler.CompileConfig(prefetch_auto=False))
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

    xt = nl.ndarray(
        (partitions, free_size), dtype=nl.float32, buffer=nl.sbuf
    )
    gt = nl.ndarray(
        (partitions, free_size), dtype=nl.float32, buffer=nl.sbuf
    )
    bt = nl.ndarray(
        (partitions, free_size), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.dma_copy(dst=xt, src=xv[first_tile])
    nisa.dma_copy(dst=gt, src=gv[first_tile])
    nisa.dma_copy(dst=bt, src=bv[first_tile])

    for i in range(tiles_per_core):
        # Queue independent reads ahead of the current tile's dependent
        # output DMA, allowing DMA and compute to overlap.
        if i + 1 < tiles_per_core:
            next_tile = first_tile + i + 1
            next_xt = nl.ndarray(
                (partitions, free_size), dtype=nl.float32, buffer=nl.sbuf
            )
            next_gt = nl.ndarray(
                (partitions, free_size), dtype=nl.float32, buffer=nl.sbuf
            )
            next_bt = nl.ndarray(
                (partitions, free_size), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.dma_copy(dst=next_xt, src=xv[next_tile])
            nisa.dma_copy(dst=next_gt, src=gv[next_tile])
            nisa.dma_copy(dst=next_bt, src=bv[next_tile])

        nisa.tensor_tensor(
            dst=gt,
            data1=xt,
            data2=gt,
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
        nisa.tensor_tensor(
            dst=gt,
            data1=gt,
            data2=bt,
            op=nl.add,
            engine=nisa.engine.vector,
        )
        nisa.activation(
            dst=gt,
            op=nl.silu,
            data=gt,
            engine=nisa.engine.scalar,
        )
        nisa.dma_copy(dst=ov[first_tile + i], src=gt)

        if i + 1 < tiles_per_core:
            xt = next_xt
            gt = next_gt
            bt = next_bt

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
    free_size = min(4096, max(1, n // (cores * partitions * 4)))
    tile_elements = partitions * free_size

    if n == 0 or n % (cores * tile_elements) != 0:
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
        "prefetch_auto": False,
        "lookahead_tiles": 1,
        "compute_storage": "inplace_gate_tile",
    }

    return _fused_activation[2](x, gate, bias, free_size)


def get_last_config() -> dict:
    return dict(_last_config)

import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _layernorm(x, weight, bias, eps, row_tile):
    batch, m, k = x.shape
    rows = batch * m
    inv_k = 1.0 / k

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    x2 = x.reshape((rows, k))
    y2 = out.reshape((rows, k))

    # Promote and broadcast the affine parameters once per physical core.
    w = nl.ndarray((row_tile, k), dtype=nl.float32, buffer=nl.sbuf)
    b = nl.ndarray((row_tile, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=w,
        src=weight.reshape((1, k)).ap(
            pattern=[[0, row_tile], [1, k]], offset=0
        ),
    )
    nisa.dma_copy(
        dst=b,
        src=bias.reshape((1, k)).ap(
            pattern=[[0, row_tile], [1, k]], offset=0
        ),
    )

    tiles = (rows + row_tile - 1) // row_tile
    pid = nl.program_id(0)
    programs = nl.num_programs()
    first = pid * tiles // programs
    last = (pid + 1) * tiles // programs

    for tile in range(first, last):
        r0 = tile * row_tile
        nr = min(row_tile, rows - r0)

        data = nl.ndarray((nr, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=data, src=x2[r0:r0 + nr, :])

        negative_mean = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=negative_mean,
            data=data,
            op=nl.add,
            axis=1,
            negate=True,
        )
        nisa.tensor_scalar(
            dst=negative_mean,
            data=negative_mean,
            op0=nl.multiply,
            operand0=inv_k,
        )

        # Fuse centering, squaring, and the variance reduction.
        squared = nl.ndarray((nr, k), dtype=nl.float32, buffer=nl.sbuf)
        variance = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=squared,
            op=nl.square,
            data=data,
            bias=negative_mean,
            reduce_op=nl.add,
            reduce_res=variance,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )
        nisa.tensor_scalar(
            dst=variance,
            data=variance,
            op0=nl.multiply,
            operand0=inv_k,
            op1=nl.add,
            operand1=eps,
        )

        rstd = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=variance)

        nisa.tensor_scalar(
            dst=data,
            data=data,
            op0=nl.add,
            operand0=negative_mean,
            op1=nl.multiply,
            operand1=rstd,
            engine=nisa.engine.vector,
        )
        nisa.tensor_tensor(
            dst=data, data1=data, data2=w[0:nr, :], op=nl.multiply
        )
        nisa.tensor_tensor(
            dst=data, data1=data, data2=b[0:nr, :], op=nl.add
        )

        # DMA performs the sole narrowing conversion at the output boundary.
        nisa.dma_copy(dst=y2[r0:r0 + nr, :], src=data)

    return out


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    global _last_config
    row_tile = 128
    rows = x.shape[0] * x.shape[1]
    k = x.shape[2]
    _last_config = {
        "lnc": 2,
        "row_tile": row_tile,
        "column_tile": k,
        "row_blocks": (rows + row_tile - 1) // row_tile,
        "statistics": "centered_fp32",
        "affine": "fp32",
        "parameter_load": "broadcast_once_per_core",
        "row_loop": "unrolled",
    }
    return _layernorm[2](x, weight, bias, float(eps), row_tile)


def get_last_config() -> dict:
    return dict(_last_config)

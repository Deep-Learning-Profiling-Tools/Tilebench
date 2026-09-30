import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_R": 2, "BLOCK_C": 128, "threads": 128}
_last_autotune_config: dict = {}


def max_pool2d_config():
    return [
        dict(BLOCK_R=br, BLOCK_C=bc, threads=nt)
        for br, bc in [(1, 128), (1, 256), (1, 512), (2, 128), (2, 256),
                       (4, 128), (4, 256), (8, 64)]
        for nt in [128, 256]
    ]


@tilelang.autotune(configs=max_pool2d_config(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def max_pool2d_kernel(input, output, N, C, H, W, kernel_size, dtype,
                      stride, padding,
                      BLOCK_R: int = 2,
                      BLOCK_C: int = 128,
                      threads: int = 128):
    planes = N * C
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    input: T.Tensor((planes, H, W), dtype)
    output: T.Tensor((planes, H_out, W_out), dtype)

    with T.Kernel(
        planes,
        T.ceildiv(H_out, BLOCK_R),
        T.ceildiv(W_out, BLOCK_C),
        threads=threads,
    ) as (plane, pid_r, pid_c):
        acc = T.alloc_fragment((BLOCK_R, BLOCK_C), dtype)
        neg_inf = -T.infinity(dtype)
        T.annotate_safe_value({input: neg_inf})
        T.fill(acc, neg_inf)

        row_start = pid_r * BLOCK_R
        col_start = pid_c * BLOCK_C
        for kh in T.unroll(kernel_size):
            for kw in T.unroll(kernel_size):
                for i, j in T.Parallel(BLOCK_R, BLOCK_C):
                    oh = row_start + i
                    ow = col_start + j
                    ih = oh * stride + kh - padding
                    iw = ow * stride + kw - padding
                    acc[i, j] = T.max(acc[i, j], input[plane, ih, iw])

        T.copy(
            acc,
            output[
                plane,
                row_start: row_start + BLOCK_R,
                col_start: col_start + BLOCK_C,
            ],
        )


def run(input, N, C, H, W, kernel_size, stride, padding,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    total_out = N * C * H_out * W_out
    dtype = str(input.dtype).removeprefix("torch.")
    if total_out <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_out, dtype=input.dtype, device=input.device)
    input_3d = input.view(N * C, H, W)
    output_3d = output.view(N * C, H_out, W_out)

    if autotune:
        with set_autotune_inputs(input_3d, output_3d):
            tuned_kernel = max_pool2d_kernel.compile(
                input_3d, output_3d, N=N, C=C, H=H, W=W,
                kernel_size=kernel_size, dtype=dtype,
                stride=stride, padding=padding,
            )
            _last_autotune_config.clear()
            _last_autotune_config.update(dict(tuned_kernel.config or {}))
            tuned_kernel(input_3d, output_3d)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        max_pool2d_kernel(
            input_3d, output_3d, N=N, C=C, H=H, W=W,
            kernel_size=kernel_size, dtype=dtype,
            stride=stride, padding=padding,
            BLOCK_R=cfg["BLOCK_R"],
            BLOCK_C=cfg["BLOCK_C"],
            threads=cfg["threads"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None

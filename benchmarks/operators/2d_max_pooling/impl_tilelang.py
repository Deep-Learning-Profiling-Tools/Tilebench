import torch
import tilelang 
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 256, "threads": 128}
_last_autotune_config: dict = {}
def max_pool2d_config():
    BLOCK_SIZE = [256, 512, 1024, 2048]
    threads = [128, 256]
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]
@tilelang.autotune(configs=max_pool2d_config(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def max_pool2d_kernel(input, output, N, C, H, W, kernel_size, dtype,
                      stride, padding,
                      BLOCK_SIZE: int = 256, 
                      threads: int = 128):
    total_in = T.const("total_in")
    total_out = T.const("total_out")
    input: T.Tensor((total_in, ), dtype)
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output: T.Tensor((total_out, ), dtype)
    
    with T.Kernel(T.ceildiv(N * C * H_out * W_out, BLOCK_SIZE), threads=threads) as pid:
        start = pid * BLOCK_SIZE
        acc = T.alloc_fragment((BLOCK_SIZE,), dtype)
        T.fill(acc, -float("inf"))
        #local_input = T.alloc_fragment((kernel_size, kernel_size), dtype)
        for i in T.Parallel(BLOCK_SIZE):
            # start is an expression o
            cur_col = (start + i) % W_out # this makes sense
            cur_row = ((start + i) // W_out) % H_out # this gives big ass column
            cur_c = ((start + i) // (W_out * H_out)) % C
            cur_n = (start + i) // (W_out * H_out * C)
            # this gives the j in [i, j] for output
            # [i, j] in output correlates to kernel on input 
            # with kernel center [stride * i, stride * j] on input
            for kh in T.unroll(kernel_size):
                for kw in T.unroll(kernel_size):
                    ih = cur_row * stride + kh - padding
                    iw = cur_col * stride + kw - padding
                    if ih >= 0 and iw >= 0 and ih < H and iw < W:
                        idx_1d = ((cur_n * C + cur_c) * H + ih) * W + iw
                        acc[i] = T.max(acc[i], T.Cast("float32", input[idx_1d]))
        T.copy(acc, output[start])

def run(input, N, C, H, W, kernel_size, stride, padding,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    total_out = N * C * H_out * W_out
    dtype = str(input.dtype).removeprefix("torch.")
    if total_out <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_out, dtype=input.dtype, device=input.device)

    if autotune:
        with set_autotune_inputs(input, output):
            tuned_kernel = max_pool2d_kernel.compile(
                input, output, N=N, C=C, H=H, W=W, kernel_size=kernel_size, dtype=dtype,
                stride=stride, padding=padding
            )
            _last_autotune_config.clear()
            _last_autotune_config.update(dict(tuned_kernel.config or {}))
            tuned_kernel(input, output)

    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        max_pool2d_kernel(
            input, output, N=N, C=C, H=H, W=W,
            kernel_size=kernel_size, dtype=dtype,
            stride=stride, padding=padding, BLOCK_SIZE=cfg["BLOCK_SIZE"],
            threads=cfg["threads"]
        )

    return output

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None


            





            


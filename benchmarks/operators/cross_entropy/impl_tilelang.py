import torch  
import tilelang  
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
#triton does nto use block_size? 
_DEFAULT_CONFIG = {"threads": 256, "num_stages": 2}
_last_autotune_config: dict = {}
def cross_entropy_config():
    threads = [32, 64, 128, 256]
    return [
        dict(threads = nt)
        for nt in threads
    ]

@tilelang.autotune(configs=cross_entropy_config(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def cross_entropy_kernel(logits, targets, output, dtype,
                         BLOCK_CLASSES : int = 1208, threads : int = 256):
    #not sure why it uses BLOCK_CLASSES naming 
    M = T.dynamic("M")
    N = T.const("N")
    logits : T.Tensor((M, N), dtype)
    targets: T.Tensor((M, ), "int64")
    output: T.Tensor((M, ), dtype)

    with T.Kernel(M, threads=threads) as row:
        local_logits = T.alloc_fragment((BLOCK_CLASSES, ), "float32")
        exp_shifted = T.alloc_fragment((BLOCK_CLASSES, ), "float32")
        row_max = T.alloc_fragment((1, ), "float32")
        row_sum = T.alloc_fragment((1, ), "float32")
        loss = T.alloc_fragment((1, ), "float32")
        T.fill(local_logits, -T.infinity("float32"))
        T.copy(logits[row, 0:N], local_logits[0:N])
        T.reduce_max(local_logits, row_max, dim = 0, clear = True)
        for i in T.Parallel(BLOCK_CLASSES):
            exp_shifted[i] = T.exp(local_logits[i] - row_max[0])
        T.reduce_sum(exp_shifted, row_sum, dim = 0, clear = True)
        #targets[row] is the index to use?
        target_lw = T.Cast("int32", targets[row])
        loss[0] = T.log(row_sum[0]) + row_max[0] - logits[row, target_lw]
        T.copy(loss, output[row])


def run(
    logits: torch.Tensor,
    targets: torch.Tensor,
    block_size: int = 1024,
    autotune: bool = False,
    **kwargs
) -> torch.Tensor:
    dtype = str(logits.dtype).removeprefix("torch.")
    batch_size, num_classes = logits.shape
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)
    grid = (batch_size,)
    block_classes = 1 << (num_classes - 1).bit_length()

    if autotune:
        with set_autotune_inputs(logits, targets, output):
            kernel = cross_entropy_kernel.compile(
                logits, targets, output,
                dtype=dtype,
                BLOCK_CLASSES=block_classes,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(logits, targets, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        cross_entropy_kernel(
            logits, targets, output,
            dtype,
            BLOCK_CLASSES=block_classes,
            threads=cfg["threads"],
        )
    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None

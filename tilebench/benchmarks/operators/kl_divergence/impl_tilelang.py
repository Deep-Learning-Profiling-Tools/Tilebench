import torch  
import tilelang  
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128, "num_stages": 3}
_last_autotune_config: dict = {}

def kl_divergence_config():
    BLOCK_SIZE = [512, 1024, 2048, 4096]
    threads = [64, 128, 256]
    num_stages = [2, 3, 4]
    return [
        dict(BLOCK_SIZE=bs, threads=nt, num_stages=ns)
        for bs in BLOCK_SIZE
        for nt in threads
        for ns in num_stages
    ]
@tilelang.autotune(configs=kl_divergence_config(), warmup = 20, rep = 100, timeout = 60)
@tilelang.jit
def kl_divergence_kernel(log_y_pred, y_true, loss, dtype,
                         BLOCK_SIZE : int = 1024, threads : int = 128, num_stages : int = 3):
    M = T.dynamic("M")
    N = T.const("N")
    log_y_pred : T.Tensor((M, N), dtype)
    y_true: T.Tensor((M, N), dtype)
    loss: T.Tensor((M, ), "float32")
    with T.Kernel(M, threads=threads) as row:
        y_true_local = T.alloc_fragment((BLOCK_SIZE, ), "float32")
        loss_local_arr = T.alloc_fragment((BLOCK_SIZE, ), "float32")
        local_log_y_pred = T.alloc_fragment((BLOCK_SIZE, ), "float32")
        loss_local = T.alloc_fragment((1, ), "float32")
        loss_global = T.alloc_fragment((1, ), "float32")
        loss_global[0] = 0.0
        for tile in T.Pipelined(T.ceildiv(N, BLOCK_SIZE), num_stages=num_stages):
            start = tile * BLOCK_SIZE
            end = T.min(start + BLOCK_SIZE, N)
            T.copy(y_true[row : row + 1, start : end], y_true_local)
            T.copy(log_y_pred[row : row + 1, start:end], local_log_y_pred)
            T.fill(loss_local_arr, 0.0)
            for i in T.Parallel(BLOCK_SIZE):
                yt = y_true_local[i]
                loss_local_arr[i] = T.Select(
                    yt > 0.0,
                    yt * (T.log(yt) - T.Cast("float32", local_log_y_pred[i])),
                    0.0,
                )
            T.reduce_sum(loss_local_arr, loss_local, dim = 0, clear = True)
            loss_global[0] += loss_local[0]
        T.copy(loss_global, loss[row:row+1])


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    dtype = str(log_y_pred.dtype).removeprefix("torch.")
    rows, cols = log_y_pred.shape
    loss = torch.empty(rows, device=log_y_pred.device, dtype=torch.float32)

    if autotune:
        with set_autotune_inputs(log_y_pred, y_true, loss):
            kernel = kl_divergence_kernel.compile(
                log_y_pred, y_true, loss,
                dtype=dtype,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(log_y_pred, y_true, loss)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        kl_divergence_kernel(
            log_y_pred, y_true, loss, dtype,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    return loss


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None

import torch
import cuda.tile as ct
import math

ConstInt = ct.Constant[int]

@ct.kernel
def matmul_kernel(
    A, B, C,
    M: ConstInt, N: ConstInt, K: ConstInt,
    TM: ConstInt, TN: ConstInt, TK: ConstInt,
    GROUP_SIZE_M: ConstInt 
):
    """
    1D Grid Launch with L2 Cache Swizzling.
    """

    pid = ct.bid(0)

    num_pid_m = (M + TM - 1) // TM
    num_pid_n = (N + TN - 1) // TN
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = GROUP_SIZE_M
    if num_pid_m - first_pid_m < GROUP_SIZE_M:
        group_size_m = num_pid_m - first_pid_m

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    acc = ct.full((TM, TN), 0.0, ct.float32)
    num_tiles_k = (K + TK - 1) // TK
    for k in range(num_tiles_k):
        a_tile = ct.load(A, (pid_m, k), (TM, TK), padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, (k, pid_n), (TK, TN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)
    acc = ct.astype(acc, C.dtype)
    ct.store(C, (pid_m, pid_n), acc)
    



def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None):
    """
    Wrapper function for cuTile matmul.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    
    M, K = a.shape
    _, N = b.shape
    dtype = a.dtype
    
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    
    if dtype == torch.float8_e4m3fn:
        TM, TN, TK = 128, 256, 128
        GROUP_SIZE_M = 8
    else: # float16
        TM, TN, TK = 128, 256, 64
        GROUP_SIZE_M = 8

    num_pid_m = math.ceil(M / TM)
    num_pid_n = math.ceil(N / TN)
    grid_1d = num_pid_m * num_pid_n
    
    grid = (grid_1d, 1, 1)
    
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        matmul_kernel,
        (a, b, c, M, N, K, TM, TN, TK, GROUP_SIZE_M)
    )
    
    return c
import torch
import cuda.tile as ct
import math

ConstInt = ct.Constant[int]

@ct.kernel
def matmul_int8_kernel(
    A, B, C,
    TM: ConstInt, TN: ConstInt, TK: ConstInt,
    GROUP_SIZE_M: ConstInt
):

    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    K_b = B.shape[0]

    pid = ct.bid(0)
    num_pid_m = ct.cdiv(M, TM)
    num_pid_n = ct.cdiv(N, TN)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    acc = ct.full((TM, TN), 0, ct.int32)
    
    num_tiles_kb = ct.cdiv(K_b, TK)
    for i in range(4):
        for j in range(num_tiles_kb):
            k_a = i * num_tiles_kb + j
            A_tile = ct.astype(ct.load(A, (pid_m, k_a), (TM, TK), padding_mode=ct.PaddingMode.ZERO), ct.int8)
            B_tile = ct.astype(ct.load(B, (j, pid_n), (TK, TN), padding_mode=ct.PaddingMode.ZERO), ct.int8)
            mask = 3 << (2 * i)
            B_tile = ct.astype(B_tile, ct.int32)
            b_unpacked = ct.astype(((B_tile & mask) >> (2 * i)), ct.int8)
            b_unpacked = b_unpacked - 1
            acc = ct.mma(A_tile, b_unpacked, acc)

    ct.store(C, index=(pid_m, pid_n), tile=acc)


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None):
    """
    Wrapper function for cuTile matmul_int8.
    """
    assert a.shape[1] == b.shape[0] * 4, "Incompatible dimensions"
    
    M, K = a.shape
    K_b, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    
    TM, TN, TK = 128, 128, 32
    GROUP_SIZE_M = 8
        
    num_pid_m = math.ceil(M / TM)
    num_pid_n = math.ceil(N / TN)
    grid_1d = num_pid_m * num_pid_n
    
    grid = (grid_1d, 1, 1)
    
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        matmul_int8_kernel,
        (a, b, c, TM, TN, TK, GROUP_SIZE_M)
    )
    
    return c
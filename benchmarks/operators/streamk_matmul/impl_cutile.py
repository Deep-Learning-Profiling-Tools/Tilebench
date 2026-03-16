import torch
import cuda.tile as ct
import math

ConstInt = ct.Constant[int]

# ==============================================================================
# Helper Functions (Inlined by cuTile trace)
# ==============================================================================
def get_swizzle_tile_coords(tile_id, M, N, TM, TN, GROUP_M):
    """
    Translates a linear tile_id into a 2D (pid_m, pid_n) coordinate
    using L2 cache swizzling.
    """
    grid_m = (M + TM - 1) // TM
    grid_n = (N + TN - 1) // TN
    width = GROUP_M * grid_n
    
    group_id = tile_id // width
    group_size = ct.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n

# ==============================================================================
# Kernels
# ==============================================================================

@ct.kernel
def first_wave_kernel(
    A, B, C, Locks,
    TM: ConstInt, TN: ConstInt, TK: ConstInt,
    total_full_tiles_streamk: ConstInt,
    total_partial_tiles_streamk: ConstInt,
    iters_per_tile: ConstInt,
    GROUP_M: ConstInt
):
    """
    The Stream-K worker kernel. It computes an assigned range of K-iterations.
    """
    # Fetch dynamic shapes to prevent loop unrolling
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]
    
    pid = ct.bid(0)
    
    # Calculate the global iteration range assigned to this specific block (SM)
    start_iter = pid * total_full_tiles_streamk + ct.minimum(pid, total_partial_tiles_streamk)
    last_iter = (pid + 1) * total_full_tiles_streamk + ct.minimum(pid + 1, total_partial_tiles_streamk)
    
    # Loop over the assigned iteration chunks
    while start_iter < last_iter:
        # Determine where this tile's iterations end
        rem = iters_per_tile - (start_iter % iters_per_tile)
        end_iter = ct.minimum(start_iter + rem, last_iter)
        
        # Calculate Tile coordinates
        tile_id = start_iter // iters_per_tile
        pid_m, pid_n = get_swizzle_tile_coords(tile_id, M, N, TM, TN, GROUP_M)
        
        # Initialize FP32 accumulator for MMA
        acc = ct.full((TM, TN), 0.0, dtype=ct.float32)
        
        # MMA Loop (Using while to avoid compile-time loop unrolling)
        current_iter = start_iter
        while current_iter < end_iter:
            k_chunk = current_iter % iters_per_tile
            
            # Load tiles safely with padding
            a_tile = ct.load(A, index=(pid_m, k_chunk), shape=(TM, TK), padding_mode=ct.PaddingMode.ZERO)
            b_tile = ct.load(B, index=(k_chunk, pid_n), shape=(TK, TN), padding_mode=ct.PaddingMode.ZERO)
            
            # Tensor Core MMA
            acc = ct.mma(a_tile, b_tile, acc)
            current_iter = current_iter + 1
            
        # Downcast accumulator to target dtype
        acc_casted = ct.astype(acc, C.dtype)
        
        # Write-back and Synchronization Logic
        if end_iter % iters_per_tile == 0:  
            # 1. This block computed the END of the tile.
            # We can safely store to C.
            ct.store(C, index=(pid_m, pid_n), tile=acc_casted)
            
            # If we didn't start at the beginning, we must unlock the spin-lock 
            # for the block that computed the partial start.
            if start_iter % iters_per_tile != 0: 
                ct.atomic_xchg(Locks, (tile_id,), 1)
        else:
            # 2. This block computed the START of a tile but didn't finish it.
            # Must wait for the finishing block to write C first.
            while ct.atomic_cas(Locks, (tile_id,), 1, 1) != 1: 
                pass
            
            # Construct broadcastable indices for Bulk Atomic Add
            # row_indices: (TM, 1)
            row_indices = ct.expand_dims(pid_m * TM + ct.arange(TM, dtype=ct.int32), 1)
            # col_indices: (1, TN)
            col_indices = ct.expand_dims(pid_n * TN + ct.arange(TN, dtype=ct.int32), 0)
            
            # Bulk atomic add (cuTile handles OOB indices automatically)
            ct.atomic_add(C, (row_indices, col_indices), acc_casted)
            
        # Move to the next chunk
        start_iter = end_iter


@ct.kernel
def full_tiles_kernel(
    A, B, C,
    TM: ConstInt, TN: ConstInt, TK: ConstInt,
    total_tiles_streamk: ConstInt,
    GROUP_M: ConstInt
):
    """
    Standard Data-Parallel GEMM for leftover full tiles.
    """
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]
    
    pid_offset = ct.bid(0)
    tile_id = pid_offset + total_tiles_streamk
    
    # 1. Map tile_id to (pid_m, pid_n)
    pid_m, pid_n = get_swizzle_tile_coords(tile_id, M, N, TM, TN, GROUP_M)
    
    # 2. Accumulator
    acc = ct.full((TM, TN), 0.0, dtype=ct.float32)
    
    # 3. K-dimension Loop (Using while to avoid unrolling)
    num_tiles_k = (K + TK - 1) // TK
    k = 0
    while k < num_tiles_k:
        a_tile = ct.load(A, index=(pid_m, k), shape=(TM, TK), padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, index=(k, pid_n), shape=(TK, TN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)
        k = k + 1
        
    # 4. Write back
    acc_casted = ct.astype(acc, C.dtype)
    ct.store(C, index=(pid_m, pid_n), tile=acc_casted)

# ==============================================================================
# Runner Wrapper
# ==============================================================================
def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None, 
        grid_programs: int = 16, # Adjust based on target GPU SM count (e.g. 108 for A100)
        BLK_M: int = 128, BLK_N: int = 128, BLK_K: int = 32, 
        two_tiles: bool = True):
    
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    
    M, K = a.shape
    _, N = b.shape
    dtype = a.dtype
    
    # Configure shapes based on dtype
    if dtype == torch.float8_e4m3fn:
        TM, TN, TK = 128, 256, 128
    else: 
        TM, TN, TK = BLK_M, BLK_N, BLK_K
        
    GROUP_M = 8
    
    # 1. CPU-side Scheduler Math
    total_blocks_M = math.ceil(M / TM)
    total_blocks_N = math.ceil(N / TN)
    iters_per_tile = math.ceil(K / TK)
    
    total_tiles = total_blocks_M * total_blocks_N
    total_programs_streamk = grid_programs

    if total_programs_streamk > 0:
        total_tiles_streamk = total_tiles % total_programs_streamk
        if two_tiles and total_tiles - total_tiles_streamk > total_programs_streamk:
            total_tiles_streamk += total_programs_streamk
            
        total_blocking_tiles = total_tiles - total_tiles_streamk
        total_iters_streamk = total_tiles_streamk * iters_per_tile
        total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
        total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk
    else:
        total_blocking_tiles = total_tiles
        total_tiles_streamk = 0
        total_full_tiles_streamk = 0
        total_partial_tiles_streamk = 0

    # 2. Allocate outputs and locks
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # Ensure locks are initialized to 0
    locks = torch.zeros((max(1, total_tiles_streamk),), device=a.device, dtype=torch.int32)
    
    stream = torch.cuda.current_stream()
    
    # 3. Launch first_wave_kernel (Stream-K)
    if total_programs_streamk > 0:
        grid_1 = (total_programs_streamk, 1, 1)
        ct.launch(
            stream, grid_1, first_wave_kernel,
            (a, b, c, locks, TM, TN, TK,
             total_full_tiles_streamk, total_partial_tiles_streamk,
             iters_per_tile, GROUP_M)
        )
        
    # 4. Launch full_tiles_kernel (Data Parallel)
    if total_blocking_tiles > 0:
        grid_2 = (total_blocking_tiles, 1, 1)
        ct.launch(
            stream, grid_2, full_tiles_kernel,
            (a, b, c, TM, TN, TK, total_tiles_streamk, GROUP_M)
        )

    return c
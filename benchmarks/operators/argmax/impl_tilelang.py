import torch 
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
_
#no argmax, primitive use reduce_max then find index
#if slow maybe implement own
def argmax_rowwise_config():
    BLOCK_SIZE=[256, 512, 1024, 2048]
    threads=[128, 256, 512]
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]
@tilelang.autotune(configs=argmax_rowwise_config(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def argmax_rowwise_kernel(X):
    M = T.dynamic("M")
    N
    with T.Kernel()




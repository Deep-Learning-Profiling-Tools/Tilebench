import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import torch
from torch_xla.core import xla_model as xm

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

#defines maximum partitions supported
PMAX = nl.tile_size.pmax

#vector_add
@nki.jit
def matrix_copy_kernel(a_input):
    num_blocks = (a_input.shape[0] + (PMAX - 1)) // PMAX

    #allocate a result tensor in HBM (move result back to HBM)
    hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)

    #Allocate sbuf space for 128 partitions of the dataset at a time
    for i in nl.affine_range(num_blocks):
        offset = i*PMAX

        mask = nl.arange(PMAX)[:, None] < (a_input.shape[0] - offset)

        a_tile = nl.load(a_input[offset : offset + PMAX, :], mask=mask)
        
        #copy the calculated addition result back into hbm in 128 partition increments
        nl.store(hbm_result_tile[offset:offset+PMAX, :], value=a_tile, mask=mask)

    return hbm_result_tile

def run(a: torch.Tensor) -> torch.Tensor:
    device = xm.xla_device()
    a = a.to(device)

    return matrix_copy_kernel(a)

if __name__ == "__main__":
    a = torch.ones((256, 10), dtype=torch.float32)
    result = run(a)
    print("compiled OK")

    """
    device = xm.xla_device()
    #test 1: exact multiple of PMAX
    a = torch.ones((256, 10), dtype=torch.float32).to(device)
    result = matrix_copy_kernel(a)
    assert torch.allclose(result, a), "test 1 failed"
    print("test 1 passed: exact multiple of PMAX")

    #test 2: non-multiple of PMAX (exercises mask)
    a = torch.ones((244, 10), dtype=torch.float32).to(device)
    result = matrix_copy_kernel(a)
    assert torch.allclose(result, a), "test 2 failed"
    print("test 2 passed: non-multiple of PMAX")

    #test 3: known values
    a = torch.full((128, 10), 7.0, dtype=torch.float32).to(device)
    result = matrix_copy_kernel(a)
    assert torch.all(result == 7.0), "test 3 failed"
    print("test 3 passed: known values")

    print("all tests passed")
    """

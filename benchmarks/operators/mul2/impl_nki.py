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
def _mul2_kernel(a_input, b_input):
    #assert a_input & b_input are the same shape
    assert a_input.shape == b_input.shape
    assert a_input.dtype == b_input.dtype

    num_blocks = (a_input.shape[0] + (PMAX - 1)) // PMAX

    #allocate a result tensor in HBM (move result back to HBM)
    hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)


    #Allocate sbuf space for partitions of the dataset at a time
    for i in nl.affine_range(num_blocks):
        offset = i*PMAX

        mask = nl.arange(PMAX)[:, None] < (a_input.shape[0] - offset)

        a_tile = nl.load(a_input[offset : offset + PMAX, :], mask=mask)
        b_tile = nl.load(b_input[offset : offset + PMAX, :], mask=mask)

        #compute addition, returns a tensor
        sbuf_result_tile = nisa.tensor_tensor(data1=a_tile, data2=b_tile, op=nl.multiply)
        
        #copy the calculated addition result back into hbm in 128 partition increments
        nl.store(hbm_result_tile[offset:offset+PMAX, :], value=sbuf_result_tile, mask=mask)

    return hbm_result_tile

def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    assert a.dtype == b.dtype, f"dtype mismatch: {a.dtype} vs {b.dtype}"

    device = xm.xla_device()
    a = a.to(device)
    b = b.to(device)

    return _mul2_kernel(a, b)

#local main for testing
if __name__ == "__main__":
    a = torch.ones((256, 10), dtype=torch.float32)
    b = torch.ones((256, 10), dtype=torch.float32)
    result = run(a, b)
    print("compiled OK")
    
    """
    device = xm.xla_device()
    #test 1: exact multiple of PMAX
    a = torch.ones((256, 10), dtype=torch.float32).to(device)
    b = torch.ones((256, 10), dtype=torch.float32).to(device)
    result = _mul2_kernel(a, b)
    assert torch.allclose(result, a * b), "test 1 failed"  # change 2: a+b -> a*b
    print("test 1 passed: exact multiple of PMAX")

    #test 2: non-multiple of PMAX
    a = torch.ones((244, 10), dtype=torch.float32).to(device)
    b = torch.ones((244, 10), dtype=torch.float32).to(device)
    result = _mul2_kernel(a, b)
    assert torch.allclose(result, a * b), "test 2 failed"
    print("test 2 passed: non-multiple of PMAX")

    #test 3: known values - change 3: 3*5=15 instead of 3+5=8
    a = torch.full((128, 10), 3.0, dtype=torch.float32).to(device)
    b = torch.full((128, 10), 5.0, dtype=torch.float32).to(device)
    result = _mul2_kernel(a, b)
    assert torch.all(result == 15.0), "test 3 failed"
    print("test 3 passed: known values 3*5=15")

    print("all tests passed")
    """
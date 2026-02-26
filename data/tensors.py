import torch

def generate_vector_add_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    y = torch.randn(n, dtype=dtype, device=device)
    return (x, y)


def generate_sin_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    return (x,)

def generate_swiglu_inputs(batch_size, ncols, dtype=torch.float32, device='cuda'):
    x = torch.randn(batch_size, ncols, dtype=dtype, device=device)
    y = torch.randn(batch_size, ncols, dtype=dtype, device=device)
    return (x, y)


def generate_dropout_inputs(n, p, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    x_keep = (torch.rand(n, device=device) > p).to(torch.int32)
    return (x, x_keep, p)


# Registry for input generators
GENERATORS = {
    "vector_add": generate_vector_add_inputs,
    "sin": generate_sin_inputs,
    "swiglu": generate_swiglu_inputs,
    "dropout": generate_dropout_inputs,
}

def get_generator(operator_name):
    if operator_name not in GENERATORS:
        raise ValueError(f"No input generator registered for operator: {operator_name}")
    return GENERATORS[operator_name]

import torch

def generate_vector_add_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    y = torch.randn(n, dtype=dtype, device=device)
    return (x, y)


def generate_sin_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    return (x,)


def generate_cross_entropy_inputs(batch_size, num_classes, dtype=torch.float32, device='cuda'):
    logits = torch.randn((batch_size, num_classes), dtype=dtype, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device, dtype=torch.int64)
    return (logits, targets)


def generate_matrix_transpose_inputs(m, n, dtype=torch.float32, device='cuda'):
    x = torch.randn((m, n), dtype=dtype, device=device)
    return (x,)


# Registry for input generators
GENERATORS = {
    "vector_add": generate_vector_add_inputs,
    "sin": generate_sin_inputs,
    "cross_entropy": generate_cross_entropy_inputs,
    "matrix_transpose": generate_matrix_transpose_inputs,
}

def get_generator(operator_name):
    if operator_name not in GENERATORS:
        raise ValueError(f"No input generator registered for operator: {operator_name}")
    return GENERATORS[operator_name]

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


def generate_cross_entropy_inputs(batch_size, num_classes, dtype=torch.float32, device='cuda'):
    logits = torch.randn((batch_size, num_classes), dtype=dtype, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device, dtype=torch.int64)
    return (logits, targets)


def generate_matrix_transpose_inputs(m, n, dtype=torch.float32, device='cuda'):
    x = torch.randn((m, n), dtype=dtype, device=device)
    return (x,)
def generate_dropout_inputs(n, p, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    x_keep = (torch.rand(n, device=device) > p).to(torch.int32)
    return (x, x_keep, p)


def generate_quantized_gemm_inputs(m, n, k, dtype=torch.float32, device='cuda'):
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((k, n), dtype=dtype, device=device)
    scale = 0.02
    a_q = torch.clamp((a / scale).round(), -127, 127).to(torch.int8)
    b_q = torch.clamp((b / scale).round(), -127, 127).to(torch.int8)
    return (a_q, b_q, scale)


def generate_streamk_scheduling_inputs(m, n, k, dtype=torch.float32, device='cuda'):
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((k, n), dtype=dtype, device=device)
    return (a, b)


def generate_packing_values_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    return (x,)


def generate_unpacking_values_inputs(n, dtype=torch.float16, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    return (x,)


def generate_divergence_metric_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    y = torch.randn(n, dtype=dtype, device=device)
    eps = 1e-5
    return (x, y, eps)


def generate_generic_fused_container_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    gate = torch.randn(n, dtype=dtype, device=device)
    bias = torch.randn(n, dtype=dtype, device=device)
    return (x, gate, bias)


# Registry for input generators
GENERATORS = {
    "vector_add": generate_vector_add_inputs,
    "sin": generate_sin_inputs,
    "cross_entropy": generate_cross_entropy_inputs,
    "matrix_transpose": generate_matrix_transpose_inputs,
    "swiglu": generate_swiglu_inputs,
    "dropout": generate_dropout_inputs,
    "quantized_gemm": generate_quantized_gemm_inputs,
    "streamk_scheduling": generate_streamk_scheduling_inputs,
    "packing_values": generate_packing_values_inputs,
    "unpacking_values": generate_unpacking_values_inputs,
    "divergence_metric": generate_divergence_metric_inputs,
    "generic_fused_container": generate_generic_fused_container_inputs,
}

def get_generator(operator_name):
    if operator_name not in GENERATORS:
        raise ValueError(f"No input generator registered for operator: {operator_name}")
    return GENERATORS[operator_name]

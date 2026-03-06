import torch
from itertools import product


def _normalize_sequence(values, name):
    # Support expr syntax: {"expr": "..."} evaluates a Python expression.
    if isinstance(values, dict) and "expr" in values:
        result = eval(values["expr"])  # noqa: S307  (internal config only)
        if not isinstance(result, (list, tuple)) or len(result) == 0:
            raise ValueError(f"case_grid expr for '{name}' must evaluate to a non-empty list.")
        return list(result)
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError(f"case_grid field '{name}' must be a non-empty list/tuple.")
    return list(values)


def _expand_case_grid(case_grid, case_defaults=None):
    if case_defaults is None:
        case_defaults = {}
    if not isinstance(case_grid, dict) or len(case_grid) == 0:
        raise ValueError("case_grid must be a non-empty mapping of parameter -> candidate list.")

    keys = list(case_grid.keys())
    value_lists = [_normalize_sequence(case_grid[k], k) for k in keys]
    cases = []
    for combo in product(*value_lists):
        case = dict(case_defaults)
        case.update(dict(zip(keys, combo)))
        cases.append(case)
    return cases


def _vector_scale_cases():
    base_sizes = [2**18, 2**20, 2**22, 2**24, 10**6 + 123]
    dtypes = ["fp16", "fp32", "bf16"]
    block_sizes = [256, 512, 1024]
    cases = []
    for n in base_sizes:
        for dtype in dtypes:
            for bs in block_sizes:
                cases.append({"n": n, "dtype": dtype, "block_size": bs})
    return cases


CASE_PRESETS = {
    "vector_scale": _vector_scale_cases,
}

def generate_vector_add_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    y = torch.randn(n, dtype=dtype, device=device)
    return (x, y)


def generate_sin_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    return (x,)


def generate_mul2_inputs(n, dtype=torch.float32, device='cuda'):
    if dtype == torch.int8:
        # Values in [-32, 32] so that ×2 stays within int8 range [-128, 127].
        x = torch.randint(-32, 33, (n,), device=device).to(torch.int8)
    else:
        x = torch.randn(n, dtype=dtype, device=device)
    return (x,)


def generate_relu_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    return (x,)


def generate_destindex_inputs(
    batch_size,
    seq_len,
    kv_nope_head_num,
    kv_rope_head_num,
    kv_nope_head_dim,
    kv_rope_head_dim,
    dtype=torch.float16,
    device='cuda',
    **kwargs,
):
    del kwargs
    total_tokens = batch_size * seq_len
    kv_nope = torch.randn(
        (total_tokens, kv_nope_head_num, kv_nope_head_dim), dtype=dtype, device=device
    )
    kv_rope = torch.randn(
        (total_tokens, kv_rope_head_num, kv_rope_head_dim), dtype=dtype, device=device
    )
    dest_loc = torch.randperm(total_tokens, device=device, dtype=torch.int64).to(torch.int32)
    o_nope = torch.randn(
        (total_tokens, kv_nope_head_num, kv_nope_head_dim), dtype=dtype, device=device
    )
    o_rope = torch.randn(
        (total_tokens, kv_rope_head_num, kv_rope_head_dim), dtype=dtype, device=device
    )
    return (kv_nope, kv_rope, dest_loc, o_nope, o_rope)


def generate_rope_inputs(batch_size, seq_len, n_heads, head_dim, dtype=torch.float32, device='cuda', **kwargs):

    q = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype, device=device)

    half_dim = head_dim // 2
    cos = torch.randn(seq_len, half_dim, dtype=dtype, device=device)
    sin = torch.randn(seq_len, half_dim, dtype=dtype, device=device)
    
    return (q, cos, sin)
def generate_softmax_inputs(n=None, shape=None, dtype=torch.float32, device='cuda'):
    if shape is None:
        if n is None:
            raise ValueError("Must provide 'n' or 'shape' for softmax inputs")

        cols = int(n**0.5)
        rows = n // cols
        shape = (rows, cols)

    x = torch.randn(*shape, dtype=dtype, device=device)
    

    return (x,)
def generate_flash_attn_inputs(batch_size, n_heads, seq_len, head_dim, dtype=torch.float16, device='cuda', **kwargs):

    q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)

    return (q.contiguous(), k.contiguous(), v.contiguous())
def generate_flash_decode_stage2_inputs(n=None, batch=2, heads=8, seq_len=4096, head_dim=128, block_seq=128, dtype=torch.float32, device='cuda', **kwargs):
    num_blocks = (seq_len + block_seq - 1) // block_seq
    
    b_seqlen = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    mid_o = torch.randn((batch, heads, num_blocks, head_dim), dtype=dtype, device=device)
    mid_o_lse = torch.randn((batch, heads, num_blocks), dtype=dtype, device=device)
    
    # Pack block_seq as scalar tensor for operator wrappers expecting tensor input.
    block_seq_tensor = torch.tensor(block_seq, dtype=torch.int32, device='cpu')
    
    return (mid_o, mid_o_lse, b_seqlen, block_seq_tensor)
GENERATORS = {
    "vector_add": generate_vector_add_inputs,
    "sin": generate_sin_inputs,
    "mul2": generate_mul2_inputs,
    "relu": generate_relu_inputs,
    "destindex": generate_destindex_inputs,
    "rope": generate_rope_inputs,
    "flash_attention": generate_flash_attn_inputs,
    "softmax": generate_softmax_inputs,
    "flash_decode": generate_flash_decode_stage2_inputs,
}


def expand_cases(operator_name, config):
    # Priority: explicit test_cases > case_grid > case_preset
    if "test_cases" in config and config["test_cases"]:
        return list(config["test_cases"])
    if "case_grid" in config:
        defaults = config.get("case_defaults", {})
        return _expand_case_grid(config["case_grid"], defaults)
    if "case_preset" in config:
        preset_name = config["case_preset"]
        if preset_name not in CASE_PRESETS:
            raise ValueError(
                f"Unknown case_preset '{preset_name}' for operator '{operator_name}'. "
                f"Available presets: {list(CASE_PRESETS.keys())}"
            )
        return CASE_PRESETS[preset_name]()
    raise ValueError(
        f"Operator '{operator_name}' config must define one of: test_cases, case_grid, case_preset."
    )


def infer_problem_size(operator_name, params):
    if "n" in params:
        return int(params["n"])
    if "shape" in params and isinstance(params["shape"], (list, tuple)) and len(params["shape"]) > 0:
        size = 1
        for dim in params["shape"]:
            size *= int(dim)
        return size
    if operator_name == "flash_attention":
        return (
            int(params.get("batch_size", 1))
            * int(params.get("n_heads", 1))
            * int(params.get("seq_len", 1))
            * int(params.get("head_dim", 1))
        )
    if operator_name == "rope":
        return (
            int(params.get("batch_size", 1))
            * int(params.get("seq_len", 1))
            * int(params.get("n_heads", 1))
            * int(params.get("head_dim", 1))
        )
    if operator_name == "destindex":
        tokens = int(params.get("batch_size", 1)) * int(params.get("seq_len", 1))
        d1 = int(params.get("kv_nope_head_num", 1)) * int(params.get("kv_nope_head_dim", 1))
        d2 = int(params.get("kv_rope_head_num", 1)) * int(params.get("kv_rope_head_dim", 1))
        return tokens * (d1 + d2)
    if operator_name == "flash_decode":
        return (
            int(params.get("batch", 1))
            * int(params.get("heads", 1))
            * int(params.get("seq_len", 1))
            * int(params.get("head_dim", 1))
        )
    # Fallback: multiply all integer-like params.
    size = 1
    used = False
    for value in params.values():
        if isinstance(value, int):
            size *= max(1, value)
            used = True
    return size if used else 0


def get_generator(operator_name):
    if operator_name not in GENERATORS:
        raise ValueError(f"No input generator registered for operator: {operator_name}")
    return GENERATORS[operator_name]

import math

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
    if dtype == torch.int8:
        # Values in [-32, 32] so that x + y stays within int8 range [-128, 127].
        x = torch.randint(-32, 33, (n,), device=device).to(torch.int8)
        y = torch.randint(-32, 33, (n,), device=device).to(torch.int8)
    else:
        x = torch.randn(n, dtype=dtype, device=device)
        y = torch.randn(n, dtype=dtype, device=device)
    return (x, y)


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


def generate_batched_matmul_inputs(BATCH, M, N=None, K=None,
                                    dtype=torch.float32, device='cuda', **kwargs):
    if N is None:
        N = M
    if K is None:
        K = M
    A = torch.randn(BATCH * M * K, dtype=dtype, device=device)
    B = torch.randn(BATCH * K * N, dtype=dtype, device=device)
    return (A, B, BATCH, M, N, K)


def generate_jacobi_stencil_2d_inputs(rows, cols=None,
                                       dtype=torch.float32, device='cuda', **kwargs):
    if cols is None:
        cols = rows
    input = torch.randn(rows, cols, dtype=dtype, device=device)
    return (input, rows, cols)


def generate_destindex_inputs(
    batch_size,
    seq_len,
    kv_nope_head_num,
    kv_rope_head_num,
    kv_nope_head_dim,
    kv_rope_head_dim,
    dtype=torch.float16,
    device='cuda',
):
    total_tokens = batch_size * seq_len

    def _rand_tensor(shape):
        if dtype == torch.int8:
            return torch.randint(-64, 65, shape, device=device).to(torch.int8)
        return torch.randn(shape, dtype=dtype, device=device)

    kv_nope  = _rand_tensor((total_tokens, kv_nope_head_num, kv_nope_head_dim))
    kv_rope  = _rand_tensor((total_tokens, kv_rope_head_num, kv_rope_head_dim))
    dest_loc = torch.randperm(total_tokens, device=device, dtype=torch.int64).to(torch.int32)
    o_nope   = _rand_tensor((total_tokens, kv_nope_head_num, kv_nope_head_dim))
    o_rope   = _rand_tensor((total_tokens, kv_rope_head_num, kv_rope_head_dim))
    return (kv_nope, kv_rope, dest_loc, o_nope, o_rope)


def generate_kl_divergence_inputs(rows, cols, dtype=torch.float32, device='cuda', **kwargs):
    # y_pred: log-probabilities (output of log_softmax over the cols axis)
    # y_true: probabilities     (output of softmax     over the cols axis)
    logits_pred = torch.randn(rows, cols, dtype=dtype, device=device)
    logits_true = torch.randn(rows, cols, dtype=dtype, device=device)
    y_pred = torch.log_softmax(logits_pred, dim=-1)
    y_true = torch.softmax(logits_true, dim=-1)
    return (y_pred, y_true)


def generate_fused_activation_inputs(n, dtype=torch.float32, device='cuda', **kwargs):
    # Fused element-wise activation: silu(x * gate + bias)
    x    = torch.randn(n, dtype=dtype, device=device)
    gate = torch.randn(n, dtype=dtype, device=device)
    bias = torch.randn(n, dtype=dtype, device=device)
    return (x, gate, bias)


def generate_quantize_global_inputs(n, dtype=torch.float32, device='cuda'):
    return (torch.randn(n, dtype=torch.float32, device=device),)


def generate_dequantize_rowwise_inputs(rows, cols, device='cuda', **kwargs):
    # bitsandbytes-style rowwise dequant: int8 input + per-row absmax (fp32)
    # output[r, c] = state_x[r] * x[r, c] / 127  (fp16 output)
    x = torch.randint(-128, 127, (rows, cols), dtype=torch.int8, device=device)
    state_x = torch.rand(rows, dtype=torch.float32, device=device) * 10.0
    return (x, state_x)


def generate_dropout_inputs(n, p=0.5, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    x_keep = torch.bernoulli(torch.full((n,), 1 - p, device=device)).to(dtype)
    return (x, x_keep, p)


def generate_swiglu_inputs(M, N, dtype=torch.float32, device='cuda'):
    x = torch.randn(M, N, dtype=dtype, device=device)
    y = torch.randn(M, N, dtype=dtype, device=device)
    return (x, y)


def generate_matrix_transpose_inputs(m, n, dtype=torch.float32, device='cuda'):
    if dtype == torch.int8:
        x = torch.randint(-64, 65, (m, n), device=device).to(torch.int8)
    else:
        x = torch.randn(m, n, dtype=dtype, device=device)
    return (x,)


def generate_rmsnorm_inputs(batch, M, K, dtype=torch.float32, device='cuda'):
    x     = torch.randn(batch, M, K, dtype=dtype, device=device)
    rms_w = torch.randn(K, dtype=dtype, device=device)
    return (x, rms_w)


def generate_rope_inputs(batch_size, seq_len, n_heads, head_dim, dtype=torch.float32, device='cuda', **kwargs):

    q = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype, device=device)

    half_dim = head_dim // 2
    cos = torch.randn(seq_len, half_dim, dtype=dtype, device=device)
    sin = torch.randn(seq_len, half_dim, dtype=dtype, device=device)

    return (q, cos, sin)


def generate_moe_topk_gating_inputs(M, E, k, dtype=torch.float32, device='cuda', **kwargs):
    logits = torch.randn(M, E, dtype=dtype, device=device)
    return (logits, M, E, k)


def generate_softmax_inputs(n_rows=None, n_cols=None, shape=None, dtype=torch.float32, device='cuda', **kwargs):
    if shape is None:
        if n_rows is not None and n_cols is not None:
            shape = (n_rows, n_cols)
        else:
            raise ValueError("Must provide either 'shape' or both 'n_rows' and 'n_cols'")
    x = torch.randn(*shape, dtype=dtype, device=device)


    return (x,)


def generate_flash_attn_inputs(batch_size, n_heads, seq_len, head_dim, dtype=torch.float16, device='cuda', **kwargs):

    q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)

    return (q.contiguous(), k.contiguous(), v.contiguous())


def generate_flash_decode_stage2_inputs(batch=2, heads=8, seq_len=4096, head_dim=128, block_seq=128, dtype=torch.float32, device='cuda', **kwargs):
    num_blocks = (seq_len + block_seq - 1) // block_seq

    b_seqlen = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    mid_o = torch.randn((batch, heads, num_blocks, head_dim), dtype=dtype, device=device)
    mid_o_lse = torch.randn((batch, heads, num_blocks), dtype=dtype, device=device)

    # Pack block_seq as scalar tensor for operator wrappers expecting tensor input.
    block_seq_tensor = torch.tensor(block_seq, dtype=torch.int32, device='cpu')

    return (mid_o, mid_o_lse, b_seqlen, block_seq_tensor)


def generate_block_sparse_attention_inputs(B=2, H=8, M=1024, D=64, H_kv=2,
                                           BLOCK_M=64, BLOCK_N=64, BLOCK_D=64, NUM_D_BLOCKS=None,
                                           dtype=torch.float16, device='cuda', **kwargs):
    """
    Generate inputs for block sparse attention.
    Creates a simple "Local Window + Causal" sparse CSR layout.
    """
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if NUM_D_BLOCKS is None:
        if D % BLOCK_D != 0:
            raise ValueError(f"D ({D}) must be divisible by BLOCK_D ({BLOCK_D})")
        NUM_D_BLOCKS = D // BLOCK_D

    if D != BLOCK_D * NUM_D_BLOCKS:
        raise ValueError(
            f"Invalid config: D={D}, BLOCK_D={BLOCK_D}, NUM_D_BLOCKS={NUM_D_BLOCKS}"
    )
    Q = torch.randn((B, H, M, D), dtype=dtype, device=device)
    K = torch.randn((B, H_kv, M, D), dtype=dtype, device=device) # N == M
    V = torch.randn((B, H_kv, M, D), dtype=dtype, device=device)

    num_layout = 1 # Shared layout for all heads
    num_rows = math.ceil(M / BLOCK_M)
    num_cols = math.ceil(M / BLOCK_N)

    layout_csr_row_stride_h = num_rows + 1
    layout_csr_col_stride_h = num_rows * num_cols # Max possible capacity

    # We build a causal local window mask
    window_blocks = 2 # Attend to current block and 2 previous blocks

    row_ptrs = []
    col_indices =[]

    current_ptr = 0
    for r in range(num_rows):
        row_ptrs.append(current_ptr)
        # Start col is max(0, r - window_blocks)
        # End col is r (inclusive, because of causal)
        start_c = max(0, r - window_blocks)
        end_c = r
        for c in range(start_c, end_c + 1):
            col_indices.append(c)
            current_ptr += 1

    row_ptrs.append(current_ptr) # Final ptr

    # Pad col_indices to required size
    col_indices = col_indices + [0] * (layout_csr_col_stride_h - len(col_indices))

    layout_csr_row_indices = torch.tensor(row_ptrs, dtype=torch.int32, device=device)
    layout_csr_col_indices = torch.tensor(col_indices, dtype=torch.int32, device=device)

    softmax_scale = 1.0 / math.sqrt(D)
    EVEN_M = (M % BLOCK_M == 0)
    EVEN_N = (M % BLOCK_N == 0)

    return (Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
            layout_csr_row_stride_h, layout_csr_col_stride_h,
            num_layout, softmax_scale, H, H_kv, M,
            BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS)

def generate_reverse_array_inputs(n, dtype, device='cuda', **kwargs):
    if dtype == torch.int8:
        x = torch.randint(-64, 65, (n,), device=device).to(torch.int8)
    else:
        x = torch.randn(n, dtype=dtype, device=device)
    return (x, n)


def generate_3d_conv_inputs(input_depth, input_rows, input_cols=None,
                            kernel_depth=3, kernel_rows=3, kernel_cols=3,
                            dtype=torch.float32, device='cuda', **kwargs):
    if input_cols is None:
        input_cols = input_rows
    input_vol = torch.randn(input_depth * input_rows * input_cols, dtype=dtype, device=device)
    kernel = torch.randn(kernel_depth * kernel_rows * kernel_cols, dtype=dtype, device=device)
    return (input_vol, kernel, input_depth, input_rows, input_cols,
            kernel_depth, kernel_rows, kernel_cols)


def generate_1d_conv_inputs(input_size, kernel_size=127, dtype=torch.float32, device='cuda', **kwargs):
    inp = torch.randn(input_size, dtype=dtype, device=device)
    kern = torch.randn(kernel_size, dtype=dtype, device=device)
    return (inp, kern, input_size, kernel_size)


def generate_cross_entropy_inputs(batch_size, num_classes, dtype=torch.float32, device='cuda', **kwargs):
    logits = torch.randn(batch_size, num_classes, dtype=dtype, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)
    return (logits, targets)


def generate_quantized_gemm_inputs(m, n, k, scale=1.0, dtype=torch.float32, device='cuda', **kwargs):
    # Input tensors are always int8 regardless of dtype; output is fp32.
    a_q = torch.randint(-64, 65, (m, k), device=device).to(torch.int8)
    b_q = torch.randint(-64, 65, (k, n), device=device).to(torch.int8)
    return (a_q, b_q, scale)


def generate_layernorm_inputs(batch, M, K, dtype=torch.float32, device='cuda', **kwargs):
    x      = torch.randn(batch, M, K, dtype=dtype, device=device)
    weight = torch.randn(K, dtype=dtype, device=device)
    bias   = torch.randn(K, dtype=dtype, device=device)
    return (x, weight, bias)


def generate_streamk_matmul_inputs(m, n, k, dtype=torch.float32, device='cuda', **kwargs):
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    return (a, b)


def generate_matmul_fp32_fp16_fp8_inputs(M, N, K, dtype=torch.float32,
                                         device='cuda', **kwargs):
    """Inputs for plain GEMM tested across fp32 / fp16 / fp8 e4m3fn / fp8 e5m2.
    fp8 dtypes don't support torch.randn directly, so generate in fp32 and cast.
    """
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        a = torch.randn(M, K, dtype=torch.float32, device=device).to(dtype)
        b = torch.randn(K, N, dtype=torch.float32, device=device).to(dtype)
    else:
        a = torch.randn(M, K, dtype=dtype, device=device)
        b = torch.randn(K, N, dtype=dtype, device=device)
    return (a, b)


def generate_matmul_int8_inputs(M, N, K, dtype=torch.int8, device='cuda', **kwargs):
    """Inputs for matmul_int8: A is int8 (M, K), B is uint8 (K_b=K/4, N) packed
    with 4 ternary-ish 2-bit fields per byte. dtype kwarg is ignored — A is
    always int8 and B is always uint8 by construction.
    """
    assert K % 4 == 0, "K must be divisible by 4 (B holds 4 fields per byte)"
    K_b = K // 4
    a = torch.randint(-64, 65, (M, K), device=device, dtype=torch.int8)
    b = torch.randint(0, 256, (K_b, N), device=device, dtype=torch.uint8)
    return (a, b)


def generate_mean_reduction_inputs(M, N, dtype=torch.float32, device='cuda', **kwargs):
    x = torch.randn(M, N, dtype=dtype, device=device)
    return (x, 1)  # always row-wise (dim=1)


def generate_argmax_inputs(M, N, dtype=torch.float32, device='cuda', **kwargs):
    x = torch.randn(M, N, dtype=dtype, device=device)
    return (x, 1)  # always row-wise (dim=1)


def generate_l2_norm_inputs(batch, M, K, eps=1e-6, dtype=torch.float32, device='cuda', **kwargs):
    x = torch.randn(batch, M, K, dtype=dtype, device=device)
    return (x, eps)


def generate_2d_conv_inputs(
    batch, in_channels, out_channels, H,
    kernel_size=3, stride=1, padding=1, groups=1,
    dtype=torch.float32, device='cuda', **kwargs,
):
    W = H  # square spatial dims
    input  = torch.randn(batch, in_channels, H, W, dtype=dtype, device=device)
    weight = torch.randn(
        out_channels, in_channels // groups, kernel_size, kernel_size,
        dtype=dtype, device=device,
    )
    # scalar params are passed through to run() as kwargs by the engine
    return (input, weight, stride, padding, groups)


def generate_2d_max_pooling_inputs(
    N, C, H, W=None,
    kernel_size=3, stride=2, padding=1,
    dtype=torch.float32, device='cuda', **kwargs,
):
    if W is None:
        W = H  # square spatial dims
    input_flat = torch.randn(N * C * H * W, dtype=dtype, device=device)
    return (input_flat, N, C, H, W, kernel_size, stride, padding)


def generate_weight_dequant_inputs(M, TILE_SIZE, dtype, N=None, device='cuda', **kwargs):
    if N is None:
        N = M
    X = torch.randn(M, N, dtype=dtype, device=device)
    S_rows = math.ceil(M / TILE_SIZE)
    S_cols = math.ceil(N / TILE_SIZE)
    S = torch.randn(S_rows, S_cols, dtype=dtype, device=device)
    return (X, S, M, N, TILE_SIZE)
def generate_linear_self_attention_inputs(
    M,
    D,
    dtype=torch.float32,
    device='cuda',
    eps=1e-6,
    **kwargs,
):
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if dtype != torch.float32:
        raise ValueError("linear_self_attention expects float32 inputs.")

    q = torch.empty((M, D), dtype=dtype, device=device).uniform_(-3.0, 3.0)
    k = torch.empty((M, D), dtype=dtype, device=device).uniform_(-3.0, 3.0)
    v = torch.empty((M, D), dtype=dtype, device=device).uniform_(-3.0, 3.0)

    return (q.contiguous(), k.contiguous(), v.contiguous(), float(eps))
def generate_top_k_selection_inputs(
    N,
    k,
    dtype=torch.float32,
    device='cuda',
    BLOCK_SIZE=1024,
    **kwargs,
):
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if dtype != torch.float32:
        raise ValueError("top_k_selection expects float32 inputs.")

    input_tensor = torch.randn((N,), dtype=dtype, device=device)

    return (
        input_tensor.contiguous(),
        int(N),
        int(k),
        int(BLOCK_SIZE),
    )
def generate_histogramming_inputs(
    N,
    num_bins,
    dtype=torch.int32,
    device='cuda',
    **kwargs,
):
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if dtype != torch.int32:
        raise ValueError("histogramming expects int32 inputs.")

    input_tensor = torch.randint(
        low=0,
        high=int(num_bins),
        size=(int(N),),
        device=device,
        dtype=torch.int32,
    )

    return (input_tensor.contiguous(), int(N), int(num_bins))


# ---------------- generators recovered from older PR branches ----------------

def generate_batch_normalization_inputs(N, C, eps=1.0e-5,
                                         dtype=torch.float32, device='cuda', **kwargs):
    input = torch.randn(N, C, dtype=dtype, device=device)
    gamma = torch.randn(C, dtype=dtype, device=device)
    beta = torch.randn(C, dtype=dtype, device=device)
    return (input, gamma, beta, N, C, eps)


def generate_bitonic_sort_inputs(n, dtype=torch.float32, device='cuda', **kwargs):
    data = torch.randn(n, dtype=dtype, device=device)
    return (data, n)


def generate_gaussian_blur_inputs(input_rows, input_cols=None,
                                  kernel_rows=3, kernel_cols=3,
                                  dtype=torch.float32, device='cuda', **kwargs):
    if input_cols is None:
        input_cols = input_rows
    input_img = torch.randn(input_rows * input_cols, dtype=dtype, device=device)
    kernel = torch.rand(kernel_rows * kernel_cols, dtype=dtype, device=device)
    kernel = kernel / kernel.sum()
    return (input_img, kernel, input_rows, input_cols, kernel_rows, kernel_cols)


def generate_interleave_inputs(n, dtype, device='cuda', **kwargs):
    if dtype == torch.int8:
        a = torch.randint(-64, 65, (n,), device=device).to(torch.int8)
        b = torch.randint(-64, 65, (n,), device=device).to(torch.int8)
    else:
        a = torch.randn(n, dtype=dtype, device=device)
        b = torch.randn(n, dtype=dtype, device=device)
    return (a, b, n)


def generate_leaky_relu_inputs(n, dtype=torch.float32, device='cuda', **kwargs):
    x = torch.randn(n, dtype=dtype, device=device)
    return (x, n)


def generate_matrix_copy_inputs(N, dtype=torch.float32, device='cuda', **kwargs):
    if dtype == torch.int8:
        A = torch.randint(-64, 65, (N, N), device=device).to(torch.int8)
    else:
        A = torch.randn(N, N, dtype=dtype, device=device)
    return (A, N)


def generate_radix_sort_inputs(n, dtype=torch.int32, device='cuda', **kwargs):
    data = torch.randint(0, 2**31, (n,), dtype=torch.int64, device=device).to(dtype)
    return (data, n)


def generate_sigmoid_inputs(n, dtype=torch.float32, device='cuda', **kwargs):
    x = torch.randn(n, dtype=dtype, device=device)
    return (x, n)


GENERATORS = {
    "vector_add": generate_vector_add_inputs,
    "mul2": generate_mul2_inputs,
    "relu": generate_relu_inputs,
    "batched_matmul": generate_batched_matmul_inputs,
    "jacobi_stencil_2d": generate_jacobi_stencil_2d_inputs,
    "kl_divergence": generate_kl_divergence_inputs,
    "fused_activation": generate_fused_activation_inputs,
    "quantize_global": generate_quantize_global_inputs,
    "dequantize_rowwise": generate_dequantize_rowwise_inputs,
    "dropout": generate_dropout_inputs,
    "swiglu": generate_swiglu_inputs,
    "matrix_transpose": generate_matrix_transpose_inputs,
    "destindex": generate_destindex_inputs,
    "rmsnorm": generate_rmsnorm_inputs,
    "rope": generate_rope_inputs,
    "flash_attention": generate_flash_attn_inputs,
    "block_sparse_attention": generate_block_sparse_attention_inputs,
    "softmax": generate_softmax_inputs,
    "moe_topk_gating": generate_moe_topk_gating_inputs,
    "flash_decode": generate_flash_decode_stage2_inputs,
    "cross_entropy": generate_cross_entropy_inputs,
    "layernorm": generate_layernorm_inputs,
    "streamk_matmul": generate_streamk_matmul_inputs,
    "2d_conv": generate_2d_conv_inputs,
    "1d_conv": generate_1d_conv_inputs,
    "reverse_array": generate_reverse_array_inputs,
    "3d_conv": generate_3d_conv_inputs,
    "2d_max_pooling": generate_2d_max_pooling_inputs,
    "l2_norm": generate_l2_norm_inputs,
    "argmax": generate_argmax_inputs,
    "mean_reduction": generate_mean_reduction_inputs,
    "weight_dequant": generate_weight_dequant_inputs,
    "linear_self_attention": generate_linear_self_attention_inputs,
    "top_k_selection": generate_top_k_selection_inputs,
    "histogramming": generate_histogramming_inputs,
    "matmul_fp32_fp16_fp8": generate_matmul_fp32_fp16_fp8_inputs,
    "matmul_int8": generate_matmul_int8_inputs,
    "sigmoid": generate_sigmoid_inputs,
    "leaky_relu": generate_leaky_relu_inputs,
    "bitonic_sort": generate_bitonic_sort_inputs,
    "radix_sort": generate_radix_sort_inputs,
    "matrix_copy": generate_matrix_copy_inputs,
    "interleave": generate_interleave_inputs,
    "gaussian_blur": generate_gaussian_blur_inputs,
    "batch_normalization": generate_batch_normalization_inputs,
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
    if operator_name == "rmsnorm":
        return (
            int(params.get("batch", 1))
            * int(params.get("M", 1))
            * int(params.get("K", 1))
        )
    if operator_name == "flash_decode":
        return (
            int(params.get("batch", 1))
            * int(params.get("heads", 1))
            * int(params.get("seq_len", 1))
            * int(params.get("head_dim", 1))
        )
    if operator_name == "block_sparse_attention":
        return (
            int(params.get("B", 1))
            * int(params.get("H", 1))
            * int(params.get("M", 1))
            * int(params.get("D", 1))
        )
    if operator_name == "softmax":
        return int(params.get("n_rows", 1)) * int(params.get("n_cols", 1))
    if operator_name == "batched_matmul":
        BATCH = int(params.get("BATCH", 1))
        M = int(params.get("M", 1))
        N = int(params.get("N", M))
        K = int(params.get("K", M))
        return 2 * BATCH * M * N * K
    if operator_name == "jacobi_stencil_2d":
        rows = int(params.get("rows", 1))
        cols = int(params.get("cols", rows))
        return rows * cols
    if operator_name == "cross_entropy":
        return int(params.get("batch_size", 1)) * int(params.get("num_classes", 1))
    if operator_name == "quantized_gemm":
        return 2 * int(params.get("m", 1)) * int(params.get("n", 1)) * int(params.get("k", 1))
    if operator_name == "layernorm":
        return int(params.get("batch", 1)) * int(params.get("M", 1)) * int(params.get("K", 1))
    if operator_name == "streamk_matmul":
        return 2 * int(params.get("m", 1)) * int(params.get("n", 1)) * int(params.get("k", 1))
    if operator_name == "matmul_int8":
        return 2 * int(params.get("M", 1)) * int(params.get("N", 1)) * int(params.get("K", 1))
    if operator_name == "matmul_fp32_fp16_fp8":
        return 2 * int(params.get("M", 1)) * int(params.get("N", 1)) * int(params.get("K", 1))
    if operator_name in ("argmax", "mean_reduction"):
        return int(params.get("M", 1)) * int(params.get("N", 1))
    if operator_name == "l2_norm":
        return int(params.get("batch", 1)) * int(params.get("M", 1)) * int(params.get("K", 1))
    if operator_name == "2d_conv":
        batch        = int(params.get("batch", 1))
        in_channels  = int(params.get("in_channels", 1))
        out_channels = int(params.get("out_channels", 1))
        H            = int(params.get("H", 1))
        kernel_size  = int(params.get("kernel_size", 3))
        stride       = int(params.get("stride", 1))
        padding      = int(params.get("padding", 1))
        groups       = int(params.get("groups", 1))
        out_H        = (H + 2 * padding - kernel_size) // stride + 1
        return 2 * batch * out_channels * out_H * out_H * (in_channels // groups) * kernel_size ** 2
    if operator_name == "1d_conv":
        return int(params.get("input_size", 1))
    if operator_name == "3d_conv":
        return (
            int(params.get("input_depth", 1))
            * int(params.get("input_rows", 1))
            * int(params.get("input_cols", params.get("input_rows", 1)))
        )
    if operator_name == "2d_max_pooling":
        N = int(params.get("N", 1))
        C = int(params.get("C", 1))
        H = int(params.get("H", 1))
        W = int(params.get("W", H))
        kernel_size = int(params.get("kernel_size", 3))
        stride = int(params.get("stride", 2))
        padding = int(params.get("padding", 1))
        H_out = (H + 2 * padding - kernel_size) // stride + 1
        W_out = (W + 2 * padding - kernel_size) // stride + 1
        return N * C * H_out * W_out
    if operator_name == "weight_dequant":
        M = int(params.get("M", 1))
        N = int(params.get("N", M))
        return M * N
    if operator_name == "linear_self_attention":
        return int(params.get("M", 1)) * int(params.get("D", 1))

    if operator_name == "top_k_selection":
        return int(params.get("N", 1))

    if operator_name == "histogramming":
        return int(params.get("N", 1))
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

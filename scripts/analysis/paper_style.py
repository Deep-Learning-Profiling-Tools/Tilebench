"""Shared palette, markers, and operator taxonomy for the EMNLP paper figures.

Colors, markers, and category/difficulty maps are copied verbatim from
tools/figures/make_evaluation_figures_v3.py (the version that produced the
figures embedded in the submitted PDF) so that regenerated figures keep the
original color scheme.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

CATEGORY: dict[str, str] = {
    "vector_add": "Point-wise", "mul2": "Point-wise", "relu": "Point-wise",
    "leaky_relu": "Point-wise", "sigmoid": "Point-wise", "dropout": "Point-wise",
    "swiglu": "Point-wise", "fused_activation": "Point-wise",
    "rope": "Point-wise", "quantize_global": "Point-wise",
    "dequantize_rowwise": "Point-wise", "weight_dequant": "Point-wise",
    "softmax": "Reduction/Norm", "layernorm": "Reduction/Norm",
    "rmsnorm": "Reduction/Norm", "l2_norm": "Reduction/Norm",
    "cross_entropy": "Reduction/Norm", "kl_divergence": "Reduction/Norm",
    "mean_reduction": "Reduction/Norm", "argmax": "Reduction/Norm",
    "histogramming": "Reduction/Norm", "moe_topk_gating": "Reduction/Norm",
    "batch_normalization": "Reduction/Norm",
    "flash_attention": "Matrix Mult./Attn", "flash_decode": "Matrix Mult./Attn",
    "block_sparse_attention": "Matrix Mult./Attn",
    "matmul_fp32_fp16_fp8": "Matrix Mult./Attn",
    "matmul_int8": "Matrix Mult./Attn",
    "streamk_matmul": "Matrix Mult./Attn",
    "batched_matmul": "Matrix Mult./Attn",
    "linear_self_attention": "Matrix Mult./Attn",
    "1d_conv": "Stencil/Conv", "2d_conv": "Stencil/Conv",
    "3d_conv": "Stencil/Conv", "2d_max_pooling": "Stencil/Conv",
    "jacobi_stencil_2d": "Stencil/Conv", "gaussian_blur": "Stencil/Conv",
    "matrix_copy": "Data Layout", "matrix_transpose": "Data Layout",
    "interleave": "Data Layout", "destindex": "Data Layout",
    "bitonic_sort": "Data Layout", "radix_sort": "Data Layout",
    "top_k_selection": "Data Layout", "reverse_array": "Data Layout",
}

DIFFICULTY: dict[str, int] = {
    "vector_add": 1, "mul2": 1, "relu": 1, "leaky_relu": 1, "sigmoid": 1,
    "dropout": 1, "swiglu": 1, "fused_activation": 1, "rope": 1,
    "quantize_global": 1, "dequantize_rowwise": 1, "weight_dequant": 1,
    "softmax": 3, "layernorm": 2, "rmsnorm": 3, "l2_norm": 3,
    "cross_entropy": 2, "kl_divergence": 3, "mean_reduction": 2,
    "argmax": 2, "histogramming": 3, "moe_topk_gating": 3,
    "batch_normalization": 3,
    "flash_attention": 5, "flash_decode": 3, "block_sparse_attention": 5,
    "matmul_fp32_fp16_fp8": 4, "matmul_int8": 4, "streamk_matmul": 5,
    "batched_matmul": 4, "linear_self_attention": 5,
    "1d_conv": 2, "2d_conv": 4, "3d_conv": 2, "2d_max_pooling": 2,
    "jacobi_stencil_2d": 2, "gaussian_blur": 2,
    "matrix_copy": 1, "matrix_transpose": 1, "interleave": 1,
    "destindex": 1, "bitonic_sort": 3, "radix_sort": 3,
    "top_k_selection": 3, "reverse_array": 1,
}

CAT_ORDER = ["Stencil/Conv", "Matrix Mult./Attn", "Reduction/Norm",
             "Point-wise", "Data Layout"]

CAT_COLOR = {
    "Stencil/Conv":       "#2ca02c",
    "Matrix Mult./Attn":  "#9467bd",
    "Reduction/Norm":     "#d62728",
    "Point-wise":         "#e8c61c",
    "Data Layout":        "#17becf",
}

CAT_MARKER = {
    "Stencil/Conv":       "o",
    "Matrix Mult./Attn":  "s",
    "Reduction/Norm":     "D",
    "Point-wise":         "^",
    "Data Layout":        "*",
}

BACKEND_COLOR = {
    "triton": "#1f77b4",
    "cutile": "#ff7f0e",
}

MODEL_MARKER = {
    "gpt-5.5": "o",
    "claude-opus-4-7": "^",
}


def legend_categories(ax, loc="lower right"):
    import matplotlib.pyplot as plt
    handles = []
    for cat in CAT_ORDER:
        handles.append(plt.Line2D(
            [0], [0], linestyle="", marker=CAT_MARKER[cat],
            markerfacecolor=CAT_COLOR[cat], markeredgecolor="black",
            markersize=8, label=cat, markeredgewidth=0.4,
        ))
    ax.legend(handles=handles, loc=loc, frameon=True, fontsize=8,
              edgecolor="lightgray", facecolor="white")

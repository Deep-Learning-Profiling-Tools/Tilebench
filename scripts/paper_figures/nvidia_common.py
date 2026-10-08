"""Shared definitions for the NVIDIA paper-figure extraction (B200, GH200). See artifacts/paper_figures/schema_v1.md."""
import hashlib
import json
import re
import subprocess

HF_REPO = "bcui2/NCU_report"
DSLS = ("triton", "cutile", "tilelang")
DSL_COLUMN = {"triton": "triton_ms", "cutile": "cutile_ms", "tilelang": "tilelang_ms"}

DEVICES = {
    "B200": {
        "architecture": "blackwell", "compute_capability": "10.0", "hf_folder": "NVIDIA_B200",
        "catalogue_ref": ("origin/archive/raw-logs-2026-09-18", "outputs/profiling/B200/ncu_catalogue.json"),
        "kernel_counts_ref": ("origin/archive/raw-logs-2026-09-18", "outputs/profiling/B200/kernel_counts.json"),
        "tilelang_autotune_ref": ("origin/archive/raw-logs-2026-09-18", "results/B200/logs/autotune_logs/{op}_tilelang_autotune.json"),
    },
    "GH200": {
        "architecture": "hopper", "compute_capability": "9.0", "hf_folder": "NVIDIA_GH200",
        "catalogue_ref": ("origin/archive/tilebenchpp-2026-10", "outputs/profiling/GH200/ncu_catalogue.json"),
        "kernel_counts_ref": ("origin/archive/tilebenchpp-2026-10", "outputs/profiling/GH200/kernel_counts.json"),
        "manifest_ref": ("origin/archive/tilebenchpp-2026-10", "outputs/profiling/GH200/ncu_sweep/report_manifest.json"),
        "provenance_ref": ("origin/archive/tilebenchpp-2026-10", "outputs/profiling/GH200/ncu_sweep/PROVENANCE.md"),
    },
}

# Operator categories as listed in the TileBench paper, Table "Operator benchmark suite"
# (TileBench_EMNLP_26 table_tex/benchmark_list.tex); 12/11/8/6/8 operators.
CATEGORY = {
    **{o: "Point-wise" for o in ["rope", "vector_add", "mul2", "relu", "quantize_global", "dequantize_rowwise",
                                  "dropout", "swiglu", "fused_activation", "weight_dequant", "leaky_relu", "sigmoid"]},
    **{o: "Reduction/Normalization" for o in ["softmax", "layernorm", "rmsnorm", "cross_entropy", "kl_divergence",
                                               "mean_reduction", "argmax", "l2_norm", "histogramming",
                                               "batch_normalization", "moe_topk_gating"]},
    **{o: "Matrix Multiplication/Attention" for o in ["flash_attention", "flash_decode", "block_sparse_attention",
                                                       "matmul_fp32_fp16_fp8", "matmul_int8", "streamk_matmul",
                                                       "linear_self_attention", "batched_matmul"]},
    **{o: "Stencil/Convolution" for o in ["2d_conv", "jacobi_stencil_2d", "2d_max_pooling", "gaussian_blur",
                                          "1d_conv", "3d_conv"]},
    **{o: "Data Layout" for o in ["destindex", "matrix_transpose", "top_k_selection", "bitonic_sort", "radix_sort",
                                  "matrix_copy", "reverse_array", "interleave"]},
}
assert len(CATEGORY) == 45

DTYPE_ALIASES = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}


def norm_dtype(d):
    return DTYPE_ALIASES.get(d, d)


def canonical_params(params):
    return json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def case_id(operator, dtype, params):
    """SHA256 over the canonical JSON of {operator, dtype, params} (sorted keys, ',' ':' separators)."""
    blob = json.dumps({"dtype": norm_dtype(dtype), "operator": operator, "params": params},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def git_show_json(repo, ref, path):
    out = subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def git_show_text(repo, ref, path):
    out = subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def git_rev(repo, ref):
    return subprocess.run(["git", "-C", repo, "rev-parse", ref], capture_output=True, text=True).stdout.strip()


def git_last_commit(repo, path):
    return subprocess.run(["git", "-C", repo, "log", "-1", "--format=%H", "--", path],
                          capture_output=True, text=True).stdout.strip()


# ---------------- SASS opcode -> instruction family ----------------
FAMILY_OPCODES = {
    "integer_address": ["IMAD", "IADD3", "IADD", "VIADD", "IMUL", "IMNMX", "VIMNMX", "IABS", "LEA", "SHF", "SHL", "SHR",
                        "LOP3", "LOP", "ISCADD", "BMSK", "POPC", "FLO", "BREV", "IDP", "VIADDMNMX", "VABSDIFF",
                        "VABSDIFF4", "SGXT", "IMNMX3", "VIMNMX3", "IMADSP", "BITEXTRACT"],
    "predicate_select": ["ISETP", "FSETP", "HSETP2", "DSETP", "PSETP", "PLOP3", "SEL", "FSEL", "P2R", "R2P", "FCHK",
                         "HSET2", "FSET", "ISET", "VOTE", "VOTEU"],
    "floating_point": ["FFMA", "FADD", "FMUL", "FMNMX", "FSWZADD", "FMNMX3", "FADD2", "FFMA2", "FMUL2", "HFMA2",
                       "HADD2", "HMUL2", "HMNMX2", "VHMNMX", "DFMA", "DADD", "DMUL", "DMNMX", "FFMA32I", "FADD32I",
                       "FMUL32I", "HFMA2_32I"],
    "special_function": ["MUFU"],
    "conversion_packing": ["F2F", "F2I", "I2F", "I2FP", "F2FP", "FRND", "I2I", "F2IP", "PRMT", "I2IP"],
    "data_movement": ["MOV", "S2R", "CS2R", "S2UR", "R2UR", "MOVM", "MOV32I", "R2B", "B2R"],
    "warp_collective": ["SHFL", "REDUX", "CREDUX", "MATCH"],
    "constant_load": ["LDC", "LDCU"],
    "global_load_store": ["LDG", "STG", "LD", "ST", "CCTL", "LDGMC", "STAS"],
    "async_copy_ldgsts": ["LDGSTS"],
    "atomic": ["ATOM", "ATOMG", "RED", "REDG", "ATOMS", "REDAS", "ATOMCAS"],
    "shared_memory": ["LDS", "STS", "LDSM", "STSM"],
    "local_memory": ["LDL", "STL"],
    "tma": ["UTMALDG", "UTMASTG", "UTMACCTL", "UBLKCP", "UTMAPF", "UTMAREDG", "UBLKRED", "UBLKPF", "UTMACMDFLUSH"],
    "legacy_mma": ["HMMA", "IMMA", "DMMA", "BMMA", "QMMA", "OMMA"],
    "wgmma": ["HGMMA", "IGMMA", "QGMMA", "BGMMA", "OGMMA"],
    "tcgen05": ["UTCHMMA", "UTCQMMA", "UTCIMMA", "UTCOMMA", "UTCMMA", "UTCBAR", "UTCCP", "UTCSHIFT", "UTCATOMSWS",
                "LDTM", "STTM", "UTCATOM"],
    "synchronization": ["BAR", "BSSY", "BSYNC", "WARPSYNC", "SYNCS", "MEMBAR", "DEPBAR", "ERRBAR", "ARRIVES",
                        "WARPGROUP", "FENCE", "UCGABAR_ARV", "UCGABAR_WAIT", "ELECT", "CGAERRBAR", "LDGDEPBAR",
                        "UCGABAR"],
    "control_flow": ["BRA", "EXIT", "RET", "CALL", "BREAK", "NOP", "YIELD", "JMP", "BPT", "BRX", "JMX", "KILL",
                     "NANOSLEEP", "UBRA", "ACQBULK", "BMOV", "PMTRIG", "ENDCOLLECTIVE", "BRXU", "JMXU"],
}
_OPC2FAM = {o: f for f, ops in FAMILY_OPCODES.items() for o in ops}


def instruction_family(opcode):
    """Family of a SASS opcode (modifiers after the first '.' are ignored)."""
    base = (opcode or "").split(".")[0].strip()
    if base in _OPC2FAM:
        return _OPC2FAM[base]
    if base.startswith("UTC"):
        return "tcgen05"
    if base.startswith("U") and len(base) > 1:
        inner = base[1:]
        if inner in _OPC2FAM and _OPC2FAM[inner] in ("integer_address", "predicate_select", "data_movement",
                                                    "constant_load", "control_flow", "conversion_packing"):
            return _OPC2FAM[inner]
        if inner.startswith("LDC"):
            return "constant_load"
        return "uniform_other"
    return "other_unknown"


def mem_width_bits(opcode_with_mod):
    """Access width of an LDG/STG/LDS/STS-like opcode-with-modifier string (bits)."""
    s = opcode_with_mod
    for tok, bits in ((".256", 256), (".128", 128), (".64", 64), (".U16", 16), (".S16", 16), (".U8", 8), (".S8", 8)):
        if tok in s:
            return bits
    return 32

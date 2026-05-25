"""Build prompts for the LLM code-generation loop.

Two prompt types:
  - initial:   first iteration, gives LLM the problem + framework guide
  - feedback:  later iterations, includes prior iter's code + harness errors

Output contract for the LLM:
  Two fenced code blocks per response, exactly:

      ```python title="impl_triton.py"
      <code>
      ```

      ```python title="impl_cutile.py"
      <code>
      ```
"""
from __future__ import annotations

import re
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]


# System prompt — loaded from tools/llm_codegen/system_prompt.md so it can
# be edited without touching Python code. The actual framework conventions
# (autotune, get_last_config, dtype handling, forbidden patterns) live in
# framework_guide.md and are injected into every user message.
SYSTEM = (Path(__file__).resolve().parent / "system_prompt.md").read_text()


def _output_instruction(backends: tuple[str, ...]) -> str:
    """Build the strict output-format instruction for the requested backends."""
    n = len(backends)
    blocks = "\n\n".join(
        f'    ```python title="impl_{b}.py"\n    # full Python file content here\n    ```'
        for b in backends
    )
    files = ", ".join(f"`impl_{b}.py`" for b in backends)
    plural_block = "block" if n == 1 else "blocks"
    plural_file = "file" if n == 1 else "files"
    return f"""## Output format (STRICT)

Return EXACTLY {n} fenced code {plural_block}, in this order, with {'this title' if n == 1 else 'these titles'}:

{blocks}

Emit ONLY the {files} {plural_file}. No other code blocks. No prose between or
after the {plural_block} beyond a 1-2 sentence summary of your approach. Do
NOT include test code, do NOT include PyTorch reference code (that file is
provided by the framework).
"""


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


_SKILL_PATHS = {
    "triton": _REPO_ROOT / ".claude" / "skills" / "triton-guide" / "SKILL.md",
    "cutile": _REPO_ROOT / ".claude" / "skills" / "cutile-guide" / "SKILL.md",
}


def _read_skill(backend: str) -> str:
    """Read .claude/skills/<backend>-guide/SKILL.md and strip Claude Code
    frontmatter (YAML block at the very top) so the file reads as a plain
    programming guide for the LLM. Returns "" if the skill file is absent.
    """
    path = _SKILL_PATHS.get(backend)
    if path is None or not path.exists():
        return ""
    raw = path.read_text()
    if raw.startswith("---"):
        end = raw.find("\n---\n", 4)
        if end != -1:
            raw = raw[end + len("\n---\n"):]
    return raw.lstrip()


_BACKEND_REF_HEADERS = {
    "triton": "# Triton (triton / triton.language) API Reference",
    "cutile": "# cuTile (cuda.tile) API Reference",
}

_BACKEND_REF_PREAMBLES = {
    "triton": (
        "Treat the API reference below as authoritative for the Triton "
        "version installed in this repo (3.6.0); do NOT use APIs from later "
        "versions you may have seen in training data. When writing "
        "`impl_triton.py`, every `tl.*` / `triton.*` symbol you use must "
        "appear in this reference.\n\n"
        "**IMPORTANT — autotune sections of this reference do NOT apply.** "
        "The reference below was written as a general Triton programming "
        "guide and discusses `triton.autotune` / `triton.Config` / "
        "`_DEFAULT_CONFIG` extensively. In THIS pipeline you must NOT use "
        "any of them — see the 'No autotune' rule in the TileBench "
        "framework conventions above. Use the reference for kernel-body "
        "syntax (tl.load, tl.store, tl.dot, masking, make_block_ptr, "
        "make_tensor_descriptor, etc.); ignore everything about cfg "
        "search / autotune wrappers."
    ),
    "cutile": (
        "The cuTile DSL is newer than Triton and likely sparse in your "
        "training data. Treat the API reference below as authoritative; do "
        "NOT invent attributes by analogy with Triton (e.g. `tl.range` has "
        "no `ct.range` equivalent --- use plain Python `for` loops). When "
        "writing `impl_cutile.py`, every `ct.*` symbol you use must appear "
        "in this reference.\n\n"
        "**IMPORTANT — autotune sections of this reference do NOT apply.** "
        "The reference below discusses `CutileAutotuner`, "
        "`ct_experimental.autotune_launch`, and `ct.tune.exhaustive_search`. "
        "In THIS pipeline you must NOT use any of them — see the 'No "
        "autotune' rule in the TileBench framework conventions above. "
        "Use the reference for kernel-body syntax (ct.load, ct.store, "
        "ct.mma, ct.bid, padding_mode, etc.); ignore everything about "
        "tuners and search spaces."
    ),
}


def _backend_ref_sections(backend: str) -> list[str]:
    """Render the API-reference block for one backend. Returns an empty list
    if the skill file is missing (so the prompt simply omits the section)."""
    skill = _read_skill(backend)
    if not skill:
        return []
    return [
        "",
        _BACKEND_REF_HEADERS[backend],
        "",
        _BACKEND_REF_PREAMBLES[backend],
        "",
        skill,
        "",
        "---",
    ]


def _problem_desc_path(op: str) -> Path:
    """Resolve the canonical problem-description path:
    benchmarks/problems/current/<op>_current.md (no fallback — only "current" is used).
    """
    return _REPO_ROOT / "benchmarks" / "problems" / "current" / f"{op}_current.md"


def build_initial_prompt(
    op: str, backends: tuple[str, ...] = ("triton", "cutile")
) -> str:
    """First-iteration prompt: framework guide + (cuTile reference if cuTile
    requested) + problem desc + config + torch impl.

    `backends` is normally both ("triton", "cutile").
    """
    framework_guide = _read(_THIS_DIR / "framework_guide.md")
    op_dir = _REPO_ROOT / "benchmarks" / "operators" / op
    problem_desc = _read(_problem_desc_path(op))
    config = _read(op_dir / "config.yaml")
    impl_torch = _read(op_dir / "impl_torch.py")

    files_to_emit = ", ".join(f"`impl_{b}.py`" for b in backends)
    sections = [
        f"# Task: implement operator `{op}` for TileBench",
        "",
        f"Follow the framework conventions below. Then implement the operator "
        f"described. You will emit {files_to_emit}.",
        "",
        "---",
        "",
        "# TileBench Framework Conventions",
        "",
        framework_guide,
        "",
        "---",
    ]
    # API references — one per active backend. Order follows the canonical
    # BACKENDS tuple so the prompt layout is deterministic.
    for b in ("triton", "cutile"):
        if b in backends:
            sections.extend(_backend_ref_sections(b))
    sections.extend([
        "",
        f"# Operator description: `{op}`",
        "",
        problem_desc if problem_desc else "(No description file found; use the PyTorch reference + config to infer semantics.)",
        "",
        "---",
        "",
        f"# `config.yaml` for `{op}`",
        "",
        "```yaml",
        config,
        "```",
        "",
        f"Available case variables: see `case_grid` and `case_defaults` above. Your kernels must work for all combinations.",
        "",
        "---",
        "",
        f"# PyTorch reference: `impl_torch.py`",
        "",
        "Your generated kernels must match this reference's outputs within the tolerances in `verify:` (or per-dtype defaults if absent).",
        "",
        "```python",
        impl_torch,
        "```",
        "",
        "---",
        "",
        _output_instruction(backends),
    ])
    return "\n".join(sections)


def build_feedback_prompt(
    op: str,
    iter_idx: int,
    prev_impl_triton: str,
    prev_impl_cutile: str,
    feedback: dict,
    history: list[dict] | None = None,
    best_so_far: dict | None = None,
    backends: tuple[str, ...] = ("triton", "cutile"),
) -> str:
    """Build the prompt for iteration N>0 — repeats the framework guide and
    focuses on what went wrong in iter N-1, plus a regression-detection
    section comparing prev iter against best-so-far.

    `feedback`     — the previous iter's feedback dict.
    `history`      — list of per-iter dicts {iter, active_backends,
                     per_backend: {<b>: {stop_score, verify_clean,
                     verify_fail_count, skipped}}, ...} for iters 0..N-1.
    `best_so_far`  — per-backend dict {<b>: {iter, stop_score, impl_<b>}}
                     of the best verify-clean iter for each backend (a
                     backend key may be absent if no clean iter exists yet).
    `backends`     — backends to regenerate this iter (the loop always
                     regenerates both Triton and cuTile).
    """
    framework_guide = _read(_THIS_DIR / "framework_guide.md")
    op_dir = _REPO_ROOT / "benchmarks" / "operators" / op
    problem_desc = _read(_problem_desc_path(op))
    config = _read(op_dir / "config.yaml")
    impl_torch = _read(op_dir / "impl_torch.py")

    fb_text = _format_feedback(feedback, backends)
    trajectory_text = _format_trajectory(history or [], best_so_far, backends)

    files_to_emit = ", ".join(f"`impl_{b}.py`" for b in backends)
    intro = (
        f"This is iteration {iter_idx} of the refinement loop. "
        "Your previous iteration's results are reported below. "
        "Read the trajectory carefully: if your last iteration regressed "
        "vs the best verify-clean iter so far, consider going back to that "
        "approach as your starting point and trying a different optimization. "
        f"Then re-emit {files_to_emit}."
    )

    sections = [
        f"# Task: improve operator `{op}` for TileBench (iteration {iter_idx})",
        "",
        intro,
        "",
        trajectory_text,
        "",
        "## Feedback from iteration " + str(iter_idx - 1),
        "",
        fb_text,
        "",
        "---",
        "",
    ]
    if "triton" in backends:
        sections.extend([
            "## Your previous `impl_triton.py` (iteration " + str(iter_idx - 1) + ")",
            "",
            "```python",
            prev_impl_triton,
            "```",
            "",
        ])
    if "cutile" in backends:
        sections.extend([
            "## Your previous `impl_cutile.py` (iteration " + str(iter_idx - 1) + ")",
            "",
            "```python",
            prev_impl_cutile,
            "```",
            "",
        ])

    # Per-backend best-so-far code blocks. Show one for each active backend
    # whose best-so-far iter is NOT the previous iter (else LLM already sees
    # it as "prev impl") and where the previous iter regressed or broke
    # verification. Gives the LLM a clean baseline to rebuild from.
    bsf_blocks: list[str] = []
    for b in backends:
        bsf = (best_so_far or {}).get(b)
        if bsf is None or bsf.get("iter") == iter_idx - 1:
            continue
        impl_src = bsf.get(f"impl_{b}", "")
        if not impl_src:
            continue
        bsf_blocks.extend([
            f"### Best-so-far `impl_{b}.py` (iter {bsf['iter']}, "
            f"stop_score={bsf.get('stop_score', 0)*100:.1f}%)",
            "",
            "Your previous iteration regressed (or broke verify) for this "
            "backend. Use this code as the starting point and apply a "
            "different optimization.",
            "",
            "```python",
            impl_src,
            "```",
            "",
        ])
    if bsf_blocks:
        sections.extend(["---", "", "## Best verify-clean code so far", ""])
        sections.extend(bsf_blocks)

    sections.extend([
        "---",
        "",
        "# Reminders: TileBench Framework Conventions",
        "",
        framework_guide,
        "",
        "---",
        "",
    ])
    # API references — one per active backend (mirrors build_initial_prompt).
    for b in ("triton", "cutile"):
        if b in backends:
            sections.extend(_backend_ref_sections(b))
            sections.append("")
    sections.extend([
        f"# Operator description: `{op}`",
        "",
        problem_desc if problem_desc else "(No description file found.)",
        "",
        "---",
        "",
        f"# `config.yaml` for `{op}`",
        "",
        "```yaml",
        config,
        "```",
        "",
        "---",
        "",
        f"# PyTorch reference: `impl_torch.py`",
        "",
        "```python",
        impl_torch,
        "```",
        "",
        "---",
        "",
        _output_instruction(backends),
    ])
    return "\n".join(sections)


def _fmt_cfg(cfg: dict | None) -> str:
    """One-line repr of a config dict, e.g. {BLOCK_M:128, num_warps:4}."""
    if not cfg:
        return "—"
    parts = []
    for k, v in cfg.items():
        parts.append(f"{k}:{v}")
    return "{" + ", ".join(parts) + "}"


def _format_trajectory(
    history: list[dict],
    best_so_far: dict | None,
    backends: tuple[str, ...] = ("triton", "cutile"),
) -> str:
    """Render a compact per-backend iteration trajectory + best-so-far summary.

    `history` entries should each contain `per_backend: {triton: {stop_score,
    verify_clean, verify_fail_count, skipped, cfg, speedup_vs_torch}, cutile:
    ...}` plus a top-level `iter` and optional `iter_total_s`. `best_so_far`
    is keyed by backend: `{triton: {iter, stop_score}, cutile: {iter,
    stop_score}}` (a backend key may be absent if no verify-clean iter
    exists yet).
    """
    if not history:
        return ""
    lines = ["## Iteration trajectory so far", ""]
    header = ["iter"]
    for b in ("triton", "cutile"):
        header.append(f"{b} cfg")
        header.append(f"{b} score")
        header.append(f"{b} speedup_vs_torch")
        header.append(f"{b} verify")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * (len(header)))
    for h in history:
        row = [str(h["iter"])]
        for b in ("triton", "cutile"):
            pb = h.get("per_backend", {}).get(b, {})
            if pb.get("skipped"):
                row.extend(["(skipped)", "(skipped)", "—", "—"])
            else:
                row.append(_fmt_cfg(pb.get("cfg")))
                row.append(f"{pb.get('stop_score', 0)*100:.1f}%")
                sp = pb.get("speedup_vs_torch")
                row.append(f"{sp:.2f}×" if isinstance(sp, (int, float)) and sp > 0 else "—")
                if pb.get("verify_clean"):
                    row.append("✓")
                else:
                    row.append(f"✗{pb.get('verify_fail_count', 0)}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "_`cfg` is the configuration your kernel actually used, as returned "
        "by `get_last_config()`. `speedup_vs_torch` is "
        "`torch_latency / kernel_latency` — values >1 mean your kernel beat "
        "the PyTorch reference; <1 means torch is still faster._"
    )
    lines.append("")

    # Best-so-far + regression notes, per still-active backend.
    last = history[-1]
    for b in backends:
        bsf = (best_so_far or {}).get(b)
        last_pb = last.get("per_backend", {}).get(b, {})
        if bsf is not None:
            lines.append(
                f"**Best verify-clean `{b}` so far: iter {bsf['iter']} "
                f"with stop_score={bsf.get('stop_score', 0)*100:.1f}%.**"
            )
            if last_pb.get("verify_clean") and last_pb.get("stop_score", 0) < bsf.get("stop_score", 0):
                delta = (bsf["stop_score"] - last_pb["stop_score"]) * 100
                lines.append(
                    f"⚠ Your `{b}` last iter (iter {last['iter']}) **regressed "
                    f"by {delta:.1f} pp** vs iter {bsf['iter']} — consider "
                    f"reverting to that approach."
                )
            elif not last_pb.get("verify_clean") and not last_pb.get("skipped"):
                lines.append(
                    f"⚠ Your `{b}` last iter (iter {last['iter']}) **broke "
                    f"verification** ({last_pb.get('verify_fail_count', 0)} "
                    f"cases). Revert to iter {bsf['iter']}'s code and apply a "
                    f"smaller, safer change."
                )
        else:
            lines.append(
                f"**No verify-clean `{b}` iteration yet — first achieve "
                f"correctness on all cases before optimising.**"
            )
    return "\n".join(lines)


def _format_feedback(
    feedback: dict, backends: tuple[str, ...] = ("triton", "cutile")
) -> str:
    """Convert the structured feedback dict into readable markdown for the LLM."""
    lines = []

    # Compile errors
    compile_errs = feedback.get("compile_errors", {})
    for backend, err in compile_errs.items():
        if err and backend in backends:
            lines.append(f"### ❌ `{backend}` failed to compile / import")
            lines.append("```")
            lines.append(err.strip())
            lines.append("```")
            lines.append("")

    # Per-case wall-clock timeouts.
    timeout_errs = feedback.get("case_timeout_errors", {})
    for backend, err in timeout_errs.items():
        if err and backend in backends:
            lines.append(f"### ⏱ `{backend}` exceeded per-case wall-clock cap")
            lines.append("```")
            lines.append(err.strip()[:3000])
            lines.append("```")
            lines.append("→ Action: your configuration likely produced a very slow kernel "
                         "(too-small tile, way too many CTAs, or a bad pipeline depth). "
                         "Pick a more aggressive tile size next iteration.")
            lines.append("")

    # Verify failures (only for backends still being generated)
    vf = [v for v in feedback.get("verify_failures", []) if v.get("backend") in backends]
    if vf:
        lines.append(f"### ❌ Verification failures ({len(vf)} cases)")
        for v in vf[:10]:
            lines.append(f"- `{v['backend']}` / dtype=`{v['dtype']}` / params={v['params']}: {v['error']}")
        if len(vf) > 10:
            lines.append(f"  ... and {len(vf) - 10} more.")
        lines.append("")

    # Per-backend roofline summary (only for active backends)
    for b in backends:
        score = feedback.get(f"stop_score_{b}", 0.0)
        rep = feedback.get(f"report_arith_mean_{b}", 0.0)
        lines.append(
            f"### `{b}` performance — mean roofline utilization = "
            f"**{score*100:.1f}%** (report_mean={rep*100:.1f}%)"
        )
    lines.append("")
    rl = [r for r in feedback.get("roofline_per_combo", []) if r.get("backend") in backends]
    if rl:
        lines.append("Per-(backend,dtype,case) detail (sorted by worst first):")
        rl_sorted = sorted(rl, key=lambda r: r.get("roofline_pct", 0.0))
        for r in rl_sorted[:20]:
            pct = r.get("roofline_pct", 0.0) * 100
            bound = r.get("bound_by", "?")
            sp = r.get("speedup_vs_torch")
            sp_str = f"{sp:.2f}× torch" if isinstance(sp, (int, float)) and sp > 0 else "no torch baseline"
            cfg_str = _fmt_cfg(r.get("cfg"))
            lines.append(
                f"- `{r['backend']}` / `{r['dtype']}` / {r['params']}: "
                f"{pct:.1f}% of roofline ({bound}), {sp_str}, cfg={cfg_str}"
            )
        if len(rl_sorted) > 20:
            lines.append(f"  ... and {len(rl_sorted) - 20} more cases.")
        lines.append("")
        lines.append(
            "→ Action: focus on the worst-performing combinations. "
            "If the case is bandwidth-bound, optimize memory coalescing / tile layout / async copies. "
            "If compute-bound, ensure Tensor Cores (tl.dot/ct.mma) and high arithmetic intensity per load."
        )

    return "\n".join(lines)


# ============================================================================
# Parsing LLM responses
# ============================================================================

_CODEBLOCK_RE = re.compile(
    r'```(?:python)?\s*(?:title=)?"?([^"\n]+?)"?\s*\n(.*?)```',
    re.DOTALL,
)


def parse_response(text: str) -> dict[str, str]:
    """Extract impl_triton.py and impl_cutile.py from the LLM response.

    Returns a dict {filename: code}. If a file is missing, the key is absent.
    """
    files = {}
    for m in _CODEBLOCK_RE.finditer(text):
        title = m.group(1).strip()
        code = m.group(2)
        # Normalize: accept "impl_triton.py", "triton", "impl_triton", etc.
        title_norm = title.lower().replace(".py", "").replace("_", "").replace("-", "")
        if "impltriton" in title_norm or title_norm == "triton":
            files["impl_triton.py"] = code
        elif "implcutile" in title_norm or title_norm == "cutile":
            files["impl_cutile.py"] = code
    return files

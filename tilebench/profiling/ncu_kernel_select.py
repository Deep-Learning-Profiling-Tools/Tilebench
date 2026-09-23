"""Hardened kernel selection + capture validation for the NCU harness.

Driving NCU's launch counter by launch ORDER alone is fragile: input-generator
kernels (randn, casts), ATen aux kernels (copy / fill / vectorized_elementwise),
and variable launch counts shift the position, so `--launch-skip/--launch-count`
can land on the WRONG kernel. Instead we select the op's OWN compute kernel(s)
by NAME via `ncu --kernel-name regex:...` (start-anchored), and AFTER the capture
we VALIDATE that every profiled kernel is one we asked for — failing loudly on a
mismatch rather than silently shipping a wrong-kernel report.

ncu_one.py and ncu_driver.py both import this so the selection logic is identical.
"""
import json
import re
import sys

from tilebench.paths import kernel_counts_path, ncu_catalogue_path


class MissingProfilingMetadataError(RuntimeError):
    """The profiling metadata of the requested GPU is absent."""


class MissingKernelCountsError(MissingProfilingMetadataError):
    """The kernel-count metadata of the requested GPU is absent."""


def load_catalogue(gpu: str) -> list:
    """The NCU catalogue of `gpu`. Sweep-max cases and autotune winners differ
    between GPUs, so a missing catalogue is an error: the catalogue of another
    GPU is never used instead."""
    path = ncu_catalogue_path(gpu)
    if not path.exists():
        raise MissingProfilingMetadataError(
            f"no NCU catalogue for {gpu} at {path}.\n"
            f"Build it from that GPU's autotune logs with:\n"
            f"    python scripts/profiling/ncu_catalogue.py --gpu {gpu}\n"
            f"The catalogue of another GPU is never used as a fallback.")
    return json.loads(path.read_text())


def load_kernel_counts(gpu: str):
    """Return ({(op, dtype, backend): count}, {(op, dtype, backend): [names]})
    probed on `gpu`.

    The NCU harness validates every capture against these counts and names, so
    running without them silently turns that validation off. Missing metadata is
    therefore an error rather than a default, and the counts of another GPU are
    never used instead: regenerate it with probe_kernel_count.py. Individual
    pairs may still be absent (a probe that errored records count=None); callers
    warn and fall back per pair.
    """
    path = kernel_counts_path(gpu)
    if not path.exists():
        raise MissingKernelCountsError(
            f"no kernel-count metadata for {gpu} at {path}.\n"
            f"NCU capture validation needs it: without it every pair would be "
            f"assumed to launch exactly one kernel and wrong-kernel reports "
            f"would not be detected.\n"
            f"Probe it on that GPU with:\n"
            f"    python scripts/profiling/probe_kernel_count.py --gpu {gpu}\n"
            f"The counts of another GPU are never used as a fallback.")
    counts, names = {}, {}
    for r in json.loads(path.read_text()):
        key = (r["op"], r["dtype"], r["backend"])
        if r.get("count") is not None:
            counts[key] = r["count"]
        if r.get("names"):
            names[key] = r["names"]
    return counts, names


def kernel_count_for(counts, key, default=1):
    """Count for one (op, dtype, backend), warning when the pair was not probed."""
    if key in counts:
        return counts[key]
    print(f"warn: no probed kernel count for {key}; assuming {default} launch(es). "
          f"Re-run probe_kernel_count.py if this pair is newly added.",
          file=sys.stderr)
    return default



# torch/ATen library kernels are mangled C++ (`void at::native::...<...>`) and
# carry regex-special chars; the op's OWN Triton/cuTile kernels are bare
# identifiers (cuTile appends a generated `_Kt...` specialization suffix).
_AUX_CHARS = " <>(),:*&{}[]"


def is_aux_kernel(name: str) -> bool:
    """True for non-op (ATen/library) kernels that must NOT be profiled."""
    return (not name) or name.startswith("void") or any(c in name for c in _AUX_CHARS)


def kernel_stems(names) -> list[str]:
    """The op's own compute-kernel name stems, cuTile's `_Kt...` suffix stripped
    so one stem matches both the Triton bare name and the cuTile variant."""
    return sorted({n.split("_Kt")[0] for n in (names or []) if not is_aux_kernel(n)})


def kernel_regex(names) -> str | None:
    """Start-anchored `ncu --kernel-name` regex selecting ONLY the op's compute
    kernel(s). The `^` anchor prevents accidentally matching an aux kernel that
    merely contains a stem as a substring. Returns None when no real kernel name
    is known (the caller must then warn — the launch-order fallback is fragile)."""
    stems = kernel_stems(names)
    if not stems:
        return None
    return "^(?:" + "|".join(re.escape(s) for s in stems) + ")"


def real_kernel_count(names) -> int:
    """Number of the op's own (non-aux) kernel launches per call."""
    return sum(1 for n in (names or []) if not is_aux_kernel(n))


_PROF_RE = re.compile(r'==PROF==\s+Profiling\s+"([^"]+)"')


def captured_kernels(ncu_output: str) -> list[str]:
    """Kernel names NCU actually profiled, parsed from its
    `==PROF== Profiling "<name>"` progress lines (which NCU writes to stdout).
    Lighter than re-importing the .ncu-rep just to read its kernel names."""
    return _PROF_RE.findall(ncu_output or "")


def _matches_stem(name: str, stems) -> bool:
    return any(name.split("_Kt")[0] == s or name.startswith(s) for s in stems)


def validate_capture(captured: list, expected_names, expected_count: int | None = None) -> tuple:
    """Confirm the capture is complete and correct.

    Success requires ALL of:
      - at least one kernel was captured;
      - every captured kernel matches an expected stem (when stems are known);
      - len(captured) == expected_count (when an expected launch count is
        known). This operates on the FULL launch sequence — repeated launches
        of the same kernel are counted individually, never deduplicated — so
        `expected 10 / captured 7` fails even if every name is valid.

    Returns (ok, problems) where problems is a list of human-readable
    diagnostics (unexpected names, count mismatch, empty capture).
    """
    problems: list[str] = []
    stems = kernel_stems(expected_names)
    if stems:
        unexpected = [nm for nm in captured if not _matches_stem(nm, stems)]
        if unexpected:
            problems.append(f"unexpected kernels: {unexpected} (expected stems {stems})")
    if not captured:
        problems.append("nothing captured")
    if expected_count is not None and len(captured) != expected_count:
        problems.append(
            f"launch-count mismatch: captured {len(captured)} != expected "
            f"{expected_count}; sequence={[n[:40] for n in captured]}")
    if not stems and expected_count is None:
        return True, []          # nothing to check against
    return (not problems), problems


def captured_from_report(rep_path) -> list | None:
    """Kernel names in result order from a .ncu-rep — the authoritative record
    of what NCU profiled (stdout ==PROF== lines can repeat per replay pass).
    Returns None if the report can't be read (caller falls back to stdout)."""
    import glob
    import sys as _sys
    for cand in sorted(glob.glob("/opt/nvidia/nsight-compute/*/extras/python"),
                       reverse=True):
        if cand not in _sys.path:
            _sys.path.append(cand)
    try:
        import ncu_report
        rep = ncu_report.load_report(str(rep_path))
        names = []
        for ri in range(rep.num_ranges()):
            rng = rep.range_by_idx(ri)
            for i in range(rng.num_actions()):
                names.append(rng.action_by_idx(i).name())
        return names
    except Exception:
        return None

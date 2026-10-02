"""Run directories for the native diagnostics.

Everything lives under ``<repo>/.local/neuron_native_diagnostics/<run_id>/`` (Git
ignored). A run directory is created exclusively; files are written with
exclusive creation or appended, never truncated, so no earlier result is ever
overwritten. Resuming a run skips a (operator, case, mode) only when a finished
record with the *same* source hash and environment hash already exists; a
changed source makes the old record stale and the case runs again (the old row
stays in the file, marked stale in the resume state).

Row validity is explicit and machine readable. Every row has a ``row_id`` (rows written
before the field existed get the SHA-256 of their stored line). ``row_status.jsonl`` is an
append-only sidecar of ``{"row_id", "valid_for_analysis": false, "invalid_reason",
"superseded_by"}`` entries; ``records()`` drops those rows unless asked for them, resume
ignores them, and ``append`` supersedes the previous valid row of the same
(operator, case_id, mode, stack) when a new one is written. ``records.jsonl`` itself is
never rewritten.

A run directory holding ``ARCHIVED_LEGACY_XLA.json`` is a frozen legacy XLA diagnostic run:
``archived`` is true, nothing may be appended, and only the legacy report is produced.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from pathlib import Path

from tilebench.neuron_diag.schema import Record
from tilebench.paths import REPO_ROOT

DIAG_ROOT = REPO_ROOT / ".local" / "neuron_native_diagnostics"

# Final states: a case in one of these is not re-run on resume (same hashes).
_FINAL = {"pass", "no_nki_impl", "import_incompatible", "unsupported_dtype_shape",
          "compile_failure", "runtime_failure", "correctness_failure", "cpu_fallback",
          "timing_unavailable", "timeout"}

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# Marker file of a frozen legacy XLA diagnostic run (see the module docstring).
ARCHIVE_MARKER = "ARCHIVED_LEGACY_XLA.json"

# Paths the diagnostics must never write into.
_PROTECTED = ("results",)


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"


class RunStore:
    def __init__(self, run_id: str, *, root: Path | str = DIAG_ROOT, resume: bool = False):
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError(f"invalid run id {run_id!r}")
        self.root = Path(root).resolve()
        for p in _PROTECTED:
            prot = (REPO_ROOT / p).resolve()
            if self.root == prot or prot in self.root.parents:
                raise ValueError(f"refusing to write diagnostics under {prot}")
        self.dir = self.root / run_id
        if resume:
            if not self.dir.is_dir():
                raise FileNotFoundError(f"no run to resume at {self.dir}")
        else:
            self.dir.parent.mkdir(parents=True, exist_ok=True)
            self.dir.mkdir(exist_ok=False)  # never reuse an existing run directory
        self.records_path = self.dir / "records.jsonl"
        self.status_path = self.dir / "row_status.jsonl"

    @property
    def archived(self) -> bool:
        return (self.dir / ARCHIVE_MARKER).is_file()

    def path(self, *parts: str) -> Path:
        p = (self.dir.joinpath(*parts)).resolve()
        if self.dir not in p.parents and p != self.dir:
            raise ValueError(f"{p} escapes the run directory")
        return p

    def write_new(self, rel: str, payload, *, binary: bool = False) -> Path:
        """Create a new file; fails if it already exists."""
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "xb" if binary else "x"
        with open(p, mode) as f:
            if binary:
                f.write(payload)
            elif isinstance(payload, str):
                f.write(payload)
            else:
                json.dump(payload, f, indent=1, sort_keys=True, default=str)
        return p

    def append(self, rec: Record, *, env_hash: str,
               supersede_reason: str = "superseded_by_newer_row_of_same_key") -> None:
        if self.archived:
            raise ValueError(f"{self.dir} is an archived legacy XLA run; start a new run id")
        row = rec.to_json()
        row["env_hash"] = env_hash
        row["recorded_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        row["row_id"] = hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:20]
        older = [r for r in self.records() if row_key(r) == row_key(row)]
        with open(self.records_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        for r in older:
            self.mark_invalid(r["row_id"], reason=supersede_reason, superseded_by=row["row_id"])

    def mark_invalid(self, row_id: str, *, reason: str, superseded_by: str | None = None,
                     note: str | None = None) -> None:
        entry = {"row_id": row_id, "valid_for_analysis": False, "invalid_reason": reason,
                 "superseded_by": superseded_by, "note": note,
                 "marked_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with open(self.status_path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    def row_status(self) -> dict[str, dict]:
        """row_id -> first invalidation entry (an invalidation is never undone)."""
        out: dict[str, dict] = {}
        if self.status_path.exists():
            with open(self.status_path) as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)
                        out.setdefault(e["row_id"], e)
        return out

    def records(self, *, include_invalid: bool = False) -> list[dict]:
        if not self.records_path.exists():
            return []
        rows = []
        with open(self.records_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r.setdefault("row_id", hashlib.sha256(line.rstrip("\n").encode()).hexdigest()[:20])
                    rows.append(r)
        status = self.row_status()
        for r in rows:
            st = status.get(r["row_id"])
            r["valid_for_analysis"] = st is None
            if st:
                r["invalid_reason"], r["superseded_by"] = st["invalid_reason"], st.get("superseded_by")
        return rows if include_invalid else [r for r in rows if r["valid_for_analysis"]]

    def resume_state(self, operator: str, case_id: int, mode: str, *, source_hash: str,
                     env_hash: str) -> str:
        """"done" (skip), "stale" (a finished row exists for other hashes) or "todo"."""
        state = "todo"
        for r in self.records():
            if (r["operator"], r["case_id"], r["mode"]) != (operator, case_id, mode):
                continue
            if r["status"] not in _FINAL:
                continue
            if r["source_hash"] == source_hash and r.get("env_hash") == env_hash:
                return "done"
            state = "stale"
        return state


def row_key(r: dict) -> tuple:
    """Identity of a measurement: one valid row per key is allowed in an analysis."""
    return (r["operator"], r["case_id"], r["mode"], r.get("stack"))

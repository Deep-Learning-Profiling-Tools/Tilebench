"""Append-only usage ledger and raw-response archive, per trajectory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def archive_response(dir_: Path, name: str, payload: Any) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    out = dir_ / f"{name}.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    return out


def ledger_totals(records: list[dict]) -> dict:
    """Sum of logical tokens over ledger rows; rows with unknown usage are
    counted and flagged, never imputed."""
    total_in = total_out = 0
    unknown_rows = 0
    for r in records:
        u = r.get("usage") or {}
        if u.get("status") == "unknown" or u.get("logical_total") is None:
            unknown_rows += 1
            continue
        total_in += int(u.get("logical_input") or 0)
        total_out += int(u.get("logical_output") or 0)
    return {"logical_input": total_in, "logical_output": total_out, "logical_total": total_in + total_out,
            "rows": len(records), "rows_with_unknown_usage": unknown_rows,
            "cost_exact": unknown_rows == 0}

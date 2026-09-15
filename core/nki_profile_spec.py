"""Canonical NKI profiling launch specification (identity + atomic persistence).

Every final NKI profiling run is described by an ``NkiProfileSpec``: a plain,
JSON-canonicalizable record of everything that affects what gets compiled and
executed — operator, case, input shapes/dtypes/strides, run kwargs, the exact
canonical autotune replay trace, source hashes, compiler environment, and the
pinned software versions. ``spec_id = SHA256(canonical JSON)`` names a private
per-spec directory; the profiled artifact and its manifest live only there.

Deliberately excluded from the identity: the random *values* of the generated
inputs (their shapes/dtypes/strides are included). Re-running the same case
regenerates different random values but the identical graph, so the identity —
and therefore manifest/artifact reuse — must not depend on them. The concrete
input bundle's SHA256 is still recorded in the manifest for audit.

This module must stay importable without torch / torch-xla / the Neuron SDK.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import tempfile
from typing import Any

SPEC_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 2  # v2: per-target artifact LISTS + runtime-trace timing

# Neuron runtime inspect facility, set by the profile worker before torch-xla is
# imported: DEVICE_PROFILE=1 makes the runtime write every NEFF it executes
# (identity: byte-identical to the compiler dump) plus one device trace per NEFF;
# SYSTEM_PROFILE=1 writes the system trace with one hardware event per
# execution (timing). The output dir is added per worker.
RUNTIME_INSPECT_ENV = {
    "NEURON_RT_INSPECT_ENABLE": "1",
    "NEURON_RT_INSPECT_DEVICE_PROFILE": "1",
    "NEURON_RT_INSPECT_SYSTEM_PROFILE": "1",
}


def canonical_json(obj: Any) -> str:
    """Deterministic canonical JSON: sorted keys, no whitespace, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class NkiProfileSpec:
    """Everything that determines what the final profile worker compiles/runs."""

    operator: str
    case_label: str
    case_params: dict
    dtype: str
    block_size: Any
    run_kwargs: dict                 # kwargs impl.run() accepts (minus autotune)
    input_specs: list                # per input: tensor {shape,dtype,stride} or {scalar}
    autotune_enabled: bool
    autotune_replay: list            # canonical winner trace from the selector
    verify_atol: Any
    verify_rtol: Any
    warmup: int
    repeat: int
    nki_enabled: bool
    operator_source_sha256: dict     # file -> sha256 (impl_nki / impl_torch)
    harness_source_sha256: dict      # file -> sha256 (this profiling harness)
    neuron_target: str
    logical_nc_config: str
    neuron_cc_flags: str
    software_versions: dict
    schema_version: int = SPEC_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def spec_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict()).encode())


def describe_inputs(inputs) -> list:
    """Serializable shape/dtype/stride description of the case inputs.

    ``inputs`` is the tuple the generator produced: tensors and/or python
    scalars, possibly nested in tuples/lists.
    """
    import torch  # deferred: keep module importable without torch

    def one(x):
        if isinstance(x, torch.Tensor):
            return {"kind": "tensor", "shape": list(x.shape),
                    "dtype": str(x.dtype), "stride": list(x.stride())}
        if isinstance(x, (tuple, list)):
            return {"kind": "seq", "items": [one(v) for v in x]}
        if isinstance(x, (type(None), bool, int, float, str)):
            return {"kind": "scalar", "value": x}
        raise TypeError(f"unsupported input element {type(x).__name__} in case inputs")

    return [one(x) for x in inputs]


def make_case_label(params: dict, dtype: str) -> str:
    """Stable, filesystem-safe case identity from swept params + dtype."""
    parts = [f"{k}{params[k]}" for k in sorted(params)]
    label = "_".join(parts + [dtype])
    return "".join(c if (c.isalnum() or c in "_-.") else "-" for c in label) or "case"


# ---------------------------------------------------------------------------
# Atomic persistence + locking
# ---------------------------------------------------------------------------

def atomic_write_bytes(path: str, data: bytes) -> None:
    """Write via temp file + os.replace so readers never see partial content."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_json(path: str, obj: Any) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode())


@contextlib.contextmanager
def spec_lock(spec_dir: str):
    """Exclusive per-spec advisory lock; two processes never partially write
    the same spec's artifacts/manifest concurrently."""
    os.makedirs(spec_dir, exist_ok=True)
    lock_path = os.path.join(spec_dir, ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def append_jsonl_locked(path: str, obj: Any) -> None:
    """Append one JSON line under an exclusive lock (no partial lines)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = canonical_json(obj) + "\n"
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line.encode())
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

"""Per-device measurement lock: candidates of one device are timed one at a
time. The lock is an fcntl file lock under outputs/llm_v2/locks/, so every
process on the host that follows the protocol serializes on it."""
from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path

from tilebench.llm.v2.orchestration.identity import LLM_V2_OUTPUT_ROOT


class DeviceBusy(RuntimeError):
    pass


@contextlib.contextmanager
def device_lock(device: str, *, timeout_s: float | None = None, root: Path | None = None):
    lock_dir = (root or LLM_V2_OUTPUT_ROOT) / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"{device}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    t0 = time.time()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if timeout_s is not None and time.time() - t0 > timeout_s:
                    raise DeviceBusy(f"device {device} is locked by another process ({path})")
                time.sleep(0.2)
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} t={time.time():.0f}\n".encode())
        yield path
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

"""Collect the part of TileLang's ``autotuner.log`` that one benchmark run wrote.

TileLang 0.1.11 (tilelang/autotuner/tuner.py, _init_logger_handlers) opens
``autotuner.log`` in the current working directory with
``logging.FileHandler(..., mode="w")`` the first time ``AutoTuner.run()`` is
called in a process, in default mode (all tunables given: "Skipping
compilation and using direct JIT") as well as in autotune mode. That truncates
whatever file was there. The handler then stays open, so every later tuning in
the same process, for any operator, is appended to the same file. Importing
tilelang does not touch it, an in-process or disk autotune-cache hit writes
nothing, and a process that never tunes leaves an older file in place. The log
lines do not name the operator.

So the file on disk is not evidence of the current run. Ownership comes from
the process instead: mark() records where this process's own FileHandler
stands before an operator runs, collect() copies the bytes it wrote since,
and only after checking that the path still names the file that handler
writes and that nothing else wrote to it. Nothing here imports tilelang or
changes how it tunes.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_TUNER_MODULE = "tilelang.autotuner.tuner"


def _file_handler() -> logging.FileHandler | None:
    """The autotuner FileHandler opened in this process, or None if TileLang's
    autotuner has not run here (then any autotuner.log on disk is not ours)."""
    tuner = sys.modules.get(_TUNER_MODULE)
    if tuner is None or not getattr(tuner, "_logger_handlers_initialized", False):
        return None
    for handler in tuner.logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.stream is not None:
            return handler
    return None


def mark() -> int:
    """Where the next run's autotuner output will start in this process's log.
    0 while TileLang has not opened it: its first use truncates the file."""
    handler = _file_handler()
    if handler is None:
        return 0
    handler.flush()
    return handler.stream.tell()


def collect(start: int, dest: Path) -> tuple[Path | None, str | None]:
    """Copy the autotuner output written by this process since ``start`` to
    ``dest`` (atomically). Returns (dest, None), or (None, reason) when there
    is nothing of this run to collect. A file left at ``dest`` by an earlier
    run is removed either way, so ``dest`` only ever holds this run's output."""
    dest = Path(dest)
    dest.unlink(missing_ok=True)
    handler = _file_handler()
    if handler is None:
        return None, "TileLang's autotuner did not run in this process"
    handler.flush()
    end = handler.stream.tell()
    ours = os.fstat(handler.stream.fileno())
    try:
        f = open(handler.baseFilename, "rb")
    except OSError as e:
        return None, f"{handler.baseFilename} is not readable ({e}); not collected"
    with f:
        on_disk = os.fstat(f.fileno())
        if (on_disk.st_dev, on_disk.st_ino) != (ours.st_dev, ours.st_ino):
            return None, f"{handler.baseFilename} was replaced or removed during the run; not collected"
        if on_disk.st_size != end or not 0 <= start <= end:
            return None, (f"{handler.baseFilename} was written by another process "
                          f"(size {on_disk.st_size}, this process at {end}); not collected")
        f.seek(start)
        data = f.read(end - start)
    if not data:
        return None, "no autotuner output in this run (cache hit)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return dest, None

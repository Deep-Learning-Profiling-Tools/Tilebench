"""Client-process telemetry for scheduling analysis.

One JSONL file per campaign process: periodic samples of the process's RSS,
CPU percentage and thread count, plus the number of provider requests in
flight / started / finished / failed and the task being worked on. Provider
rate-limit headers are archived separately with each transport event
(transport.jsonl). None of this is ever part of a prompt, a feedback text or
a score; it does not change any request."""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from pathlib import Path


def append_locked(path: Path, row: dict) -> None:
    """Append one JSON line under an exclusive flock (several campaign
    processes may share the file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, default=str) + "\n"
    with open(path, "a") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(line)
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class ProcessTelemetry:
    def __init__(self, path: Path, *, interval_s: float = 60.0, labels: dict | None = None):
        self.path = path
        self.interval_s = interval_s
        self.labels = dict(labels or {})
        self.active = 0
        self.started = 0
        self.finished = 0
        self.failed = 0
        self.active_tasks: set[str] = set()
        self.extra: dict = {}                 # scheduler state (concurrency, pending evaluations, ...)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        try:
            import psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)
        except Exception:  # noqa: BLE001 - telemetry must never break a campaign
            self._proc = None
        self._thread = threading.Thread(target=self._loop, name="tilebench-telemetry", daemon=True)

    def start(self) -> "ProcessTelemetry":
        self.sample("start")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.sample("stop")

    def task_started(self, key: str) -> None:
        with self._lock:
            self.active_tasks.add(key)

    def task_finished(self, key: str) -> None:
        with self._lock:
            self.active_tasks.discard(key)

    def begin_request(self) -> None:
        with self._lock:
            self.active += 1
            self.started += 1
        self.sample("request_start")

    def end_request(self, ok: bool) -> None:
        with self._lock:
            self.active -= 1
            self.finished += 1
            if not ok:
                self.failed += 1
        self.sample("request_end" if ok else "request_failed")

    def sample(self, event: str = "sample") -> None:
        row = {"t": time.time(), "event": event, "pid": os.getpid(), **self.labels,
               "active_requests": self.active, "requests_started": self.started,
               "requests_finished": self.finished, "requests_failed": self.failed,
               "active_trajectories": len(self.active_tasks), **{k: (v() if callable(v) else v) for k, v in self.extra.items()}}
        if self._proc is not None:
            try:
                row["rss_mb"] = round(self._proc.memory_info().rss / 2 ** 20, 1)
                row["cpu_percent"] = self._proc.cpu_percent(None)
                row["num_threads"] = self._proc.num_threads()
            except Exception:  # noqa: BLE001
                pass
        try:
            append_locked(self.path, row)
        except OSError:
            pass

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()


class TelemetryProvider:
    """Wraps a provider and counts requests in flight. The request and the
    result pass through unchanged."""

    def __init__(self, inner, telemetry: ProcessTelemetry):
        self.inner = inner
        self.telemetry = telemetry
        self.name = getattr(inner, "name", "provider")
        self.usage_schema = getattr(inner, "usage_schema", None)

    def generate(self, request):
        self.telemetry.begin_request()
        try:
            result = self.inner.generate(request)
        except BaseException:
            self.telemetry.end_request(ok=False)
            raise
        self.telemetry.end_request(ok=True)
        return result

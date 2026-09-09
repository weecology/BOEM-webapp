"""Structured, append-only run logging for prepare.py.

The bash cron wrapper (prepare_data.sh) previously overwrote its log file
on every run (`echo ... > "$LOG_FILE"`), so only the most recent run was
ever visible — there was no way to tell when prepare.py had last succeeded,
what it processed, or what it silently skipped.

This module writes one JSON object per run, appended to
app/data/logs/prepare_runs.jsonl, so run history accumulates and can be
read back later (by a person or an agent) without re-running anything.
Use analyze_runs.py to summarize it.

Usage:
    from app.utils.run_log import RunLog

    run = RunLog()
    with run.step("comet_download"):
        ...
    run.counts["crops_downloaded_experiments"] = 3
    run.finish("success")
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import traceback
import uuid

LOG_DIR = Path("app/data/logs")
LOG_PATH = LOG_DIR / "prepare_runs.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunLog:
    """Accumulates structured facts about one prepare.py run and appends
    a single JSON line to LOG_PATH when finished (success, warning, or
    failure — finish()/fail() are always called, even on exception, if the
    caller uses `with RunLog() as run:`).
    """

    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = Path(log_path)
        self.run_id = uuid.uuid4().hex[:12]
        self.started_at = _now_iso()
        self._start_perf = time.perf_counter()
        self.steps: list[dict] = []
        self.counts: dict = {}
        self.warnings: list[str] = []
        self.status = "in_progress"
        self.error: dict | None = None
        self._finished = False

    @contextmanager
    def step(self, name: str):
        """Record timing/outcome of one pipeline step. Re-raises on error
        after recording it, so the caller's own try/except (or the run's
        __exit__) still sees the exception."""
        entry = {"step": name, "started_at": _now_iso(), "status": "ok"}
        t0 = time.perf_counter()
        try:
            yield entry
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry["duration_s"] = round(time.perf_counter() - t0, 3)
            self.steps.append(entry)

    def warn(self, message: str):
        print(f"WARNING: {message}")
        self.warnings.append(message)

    def finish(self, status: str = "success"):
        if self._finished:
            return
        self._finished = True
        self.status = status
        self._write()

    def fail(self, exc: BaseException):
        if self._finished:
            return
        self._finished = True
        self.status = "failed"
        self.error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        self._write()

    def _write(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": _now_iso(),
            "duration_s": round(time.perf_counter() - self._start_perf, 3),
            "status": self.status,
            "steps": self.steps,
            "counts": self.counts,
            "warnings": self.warnings,
            "error": self.error,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"\n[run_log] run {self.run_id} recorded as '{self.status}' "
              f"in {self.log_path}")

    # Context-manager convenience: `with RunLog() as run:` auto-finishes on
    # a clean exit, or auto-fails (and re-raises) on exception.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.fail(exc)
            return False  # don't swallow the exception
        self.finish("success" if not self.warnings else "warning")
        return False

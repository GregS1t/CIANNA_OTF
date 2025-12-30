# utils/job_logger.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from datetime import datetime, timezone
import json
import logging
import os
import shutil
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# --------------------------- Types & const ---------------------------

PHASE_PENDING   = "PENDING"
PHASE_EXECUTING = "EXECUTING"
PHASE_COMPLETED = "COMPLETED"
PHASE_ERROR     = "ERROR"
PHASE_ABORTED   = "ABORTED"


ISO = "%Y-%m-%dT%H:%M:%S%z"


@dataclass
class JobRecord:
    jobId: str
    imagePath: str
    model: str = ""
    quantization: str = ""
    priority: int = 0
    status: str = PHASE_PENDING   # legacy naming kept
    phase: str  = PHASE_PENDING
    receptionDate: str = ""       
    end_time: str = ""            
    comment: str = ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data) -> None:
    path = Path(path)
    # S'assure que le dossier parent existe toujours
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


# ----------------------------- Store --------------------------------

class JobStore:
    """
    Thread-safe job logger and folder mover.
    """

    def __init__(
        self,
        logs_root: Path,
        jobs_root: Path,
        *,
        log_filename: str = "job_log.json",
        backups_dirname: str = "log_backups",
        max_backups: int = 10,
    ) -> None:
        
        self.logs_root = Path(logs_root)
        self.jobs_root = Path(jobs_root)
        self.log_path = self.logs_root / log_filename
        self.backups_dir = self.logs_root / backups_dirname
        self.max_backups = max_backups

        self._lock = Lock()
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

        self._priority_counter = 0  # will be advanced on creates

    # -------- JSON R/W (private) --------

    def _load(self) -> Dict[str, Any]:
        if not self.log_path.exists():
            return {}
        try:
            return json.loads(self.log_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read job log at %s; returning empty.", self.log_path)
            return {}

    def _backup(self, data: Dict[str, Any]) -> None:
        ts = now_iso().replace(":", "-")
        bpath = self.backups_dir / f"job_log_{ts}.json"
        try:
            bpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("Backup write failed at %s", bpath)
            return

        try:
            files = sorted(self.backups_dir.glob("job_log_*.json"))
            excess = len(files) - self.max_backups
            for p in files[:max(0, excess)]:
                p.unlink(missing_ok=True)
        except Exception:
            logger.exception("Backup rotation failed in %s", self.backups_dir)

    def _save(self, data: Dict[str, Any]) -> None:
        _atomic_write_json(self.log_path, data)
        self._backup(data)

    # -------- Public API --------

    def create(self, process_id: str, fits_name: str, *, model: str = "", quantization: str = "") -> None:
        """Create a fresh JobRecord for a new job."""
        with self._lock:
            data = self._load()
            self._priority_counter += 1
            rec = JobRecord(
                jobId=process_id,
                imagePath=fits_name,
                model=model,
                quantization=quantization,
                priority=self._priority_counter,
                status=PHASE_PENDING,
                phase=PHASE_PENDING,
                receptionDate=now_iso(),
                comment="",
            )
            data[process_id] = asdict(rec)
            self._save(data)

    def update(self, process_id: str, **kv: Any) -> None:
        """Update any fields of an existing job."""
        with self._lock:
            data = self._load()
            job = data.get(process_id)
            if not job:
                logger.warning("Job %s not found in log for update.", process_id)
                return
            for k, v in kv.items():
                if v is not None:
                    job[k] = v
            self._save(data)

    def set_phase(self, process_id: str, phase: str, *, comment: str = "") -> None:
        """Convenience to set both status/phase + optional comment."""
        with self._lock:
            data = self._load()
            job = data.get(process_id)
            if not job:
                logger.warning("Job %s not found in log for phase set.", process_id)
                return
            job["phase"] = phase
            job["status"] = phase  # keep legacy mirror
            if comment:
                job["comment"] = comment
            if phase in (PHASE_COMPLETED, PHASE_ERROR, PHASE_ABORTED):
                job["end_time"] = now_iso()
            self._save(data)

    # -------- Folder moves (JOBS tree) --------

    def _dir(self, phase: str, process_id: str) -> Path:
        return self.jobs_root / phase / process_id

    def move_to(self, process_id: str, dst_phase: str) -> None:
        """
        Move job folder to dst phase. Accepts src in PENDING or EXECUTING.
        Creates dst tree if needed.
        """
        src_exec = self._dir("EXECUTING", process_id)
        src_pend = self._dir("PENDING", process_id)
        dst_dir = self._dir(dst_phase, process_id)
        dst_dir.parent.mkdir(parents=True, exist_ok=True)

        src = src_exec if src_exec.exists() else src_pend if src_pend.exists() else None
        if src is None:
            logger.warning("No job folder found to move for %s", process_id)
            return
        try:
            src.rename(dst_dir)
        except OSError:
            shutil.move(str(src), str(dst_dir))

    def move_to_completed(self, process_id: str) -> None:
        self.move_to(process_id, "COMPLETED")
        self.set_phase(process_id, PHASE_COMPLETED)

    def move_to_error(self, process_id: str, *, comment: str = "") -> None:
        self.move_to(process_id, "ERROR")
        self.set_phase(process_id, PHASE_ERROR, comment=comment)


def default_store() -> JobStore:
    """
    Build a default JobStore using your runtime layout:
      runtime/
        JOBS/{PENDING,EXECUTING,COMPLETED,ERROR, ABORTED}/
      logs/
        job_log.json + log_backups/
    """
    runtime = Path("runtime")
    jobs_root = runtime / "JOBS"
    logs_root = Path("logs")
    logs_root.mkdir(parents=True, exist_ok=True)
    (jobs_root / "PENDING").mkdir(parents=True, exist_ok=True)
    (jobs_root / "EXECUTING").mkdir(parents=True, exist_ok=True)
    (jobs_root / "COMPLETED").mkdir(parents=True, exist_ok=True)
    (jobs_root / "ERROR").mkdir(parents=True, exist_ok=True)
    (jobs_root / "ABORTED").mkdir(parents=True, exist_ok=True)
    return JobStore(logs_root=logs_root, jobs_root=jobs_root)

# -------------------------------------------------------------------
# Module-level singleton (no circular import, usable everywhere)
# -------------------------------------------------------------------
JOB_STORE = default_store()

# --- Compatibility wrappers for legacy imports (pipeline, etc.) ----
def log_new_job(process_id: str, fits_name: str, reception_date: str) -> None:
    # create + force receptionDate to the provided value for backward-compat
    JOB_STORE.create(process_id, fits_name)
    JOB_STORE.update(process_id, receptionDate=reception_date)

def update_job_status(process_id: str, **kv) -> None:
    JOB_STORE.update(process_id, **kv)

def move_to_completed(process_id: str) -> None:
    JOB_STORE.move_to_completed(process_id)

def move_to_error(process_id: str, comment: str = "") -> None:
    JOB_STORE.move_to_error(process_id, comment=comment)

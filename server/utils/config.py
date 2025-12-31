import os
import toml
from pathlib import Path



def ensure_runtime_dirs(root="runtime"):
    root = Path(root)
    for p in (
        root,
        root / "JOBS" / "QUEUED",
        root / "JOBS" / "PENDING",
        root / "JOBS" / "EXECUTING",
        root / "JOBS" / "COMPLETED",
        root / "JOBS" / "ERROR",
        root / "JOBS" / "ABORTED"
    ):
        p.mkdir(parents=True, exist_ok=True)

class ServerConfig:
    def __init__(self, config_path="params/server.toml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.project_root = Path(__file__).resolve().parent.parent
        self._config = toml.load(config_path)
        self.verbose = self._config.get("server", {}).get("verbose", True)
        # Raw sections
        self.server = self._config.get("server", {})
        self.cianna = self._config.get("cianna", {})
        self.paths = self._config.get("paths", {})
        self.watcher = self._config.get("watcher", {})
        self.image = self._config.get("image", {})
        self.patch = self._config.get("patch", {})
        self.logging = self._config.get("logging", {})

        # --- Base directories ---
        self.project_root = Path(__file__).resolve().parent.parent
        self.runtime_root = self.project_root / self.paths.get("runtime_root", "runtime")
        self.logs_root = self.project_root / self.paths.get("logs_root", "logs")

        # --- Static paths (config, models, etc.) ---
        self.params_dir = os.path.join(self.project_root, "params")
        self.models_dir = os.path.join(self.project_root, "models")
        self.model_registry_path = os.path.join(self.models_dir,
                    self.cianna.get("model_registry", "CIANNA_models.xml"))
        self.model_save_dir = os.path.join(self.runtime_root, "net_save")

        # --- Runtime paths ---
        self.jobs_root = os.path.join(self.runtime_root, "JOBS")
        self.jobs_queued = os.path.join(self.jobs_root, "QUEUED")
        self.jobs_pending = os.path.join(self.jobs_root, "PENDING")
        self.jobs_executing = os.path.join(self.jobs_root, "EXECUTING")
        self.jobs_completed = os.path.join(self.jobs_root, "COMPLETED")
        self.jobs_error = os.path.join(self.jobs_root, "ERROR")
        self.jobs_aborted = os.path.join(self.jobs_root, "ABORTED")

        self.log_dir = self.logs_root
        self.tmp_dir = os.path.join(self.runtime_root, "tmp")

        # Scheduler settings
        self.poll_interval = self._config.get("scheduler", {}).get("poll_interval", 2.0)
        self.max_wait_time = self._config.get("scheduler", {}).get("max_wait_time", 10.0)
        self.batch_jobs_thres = self._config.get("scheduler", {}).get("batch_jobs_thres", 4)


    def get(self, section, key, default=None):
        return self._config.get(section, {}).get(key, default)

cfg = ServerConfig()
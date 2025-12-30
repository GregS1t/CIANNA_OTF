import os
import time
import shutil
from utils.config import ServerConfig
from utils.job_logger import move_to_error
from utils.job_logger import load_job_log
from utils.model_job_buffer import ModelJobBuffer
from fileio.process_xml import parse_job_parameters
from utils.config import cfg

import logging
logger = logging.getLogger(__name__)


cfg = ServerConfig()
PENDING_DIR = cfg.get("runtime", "jobs_pending", "server/runtime/JOBS/PENDING")
JOB_PENDING = cfg.jobs_pending
JOB_EXECUTING = cfg.jobs_executing


def get_jobs_by_model(pending_dir):
    """
    Regroupe les jobs en attente par nom de fichier modèle.

    Args:
        pending_dir (str): Chemin vers le dossier JOBS/PENDING

    Returns:
        dict: dictionnaire { model_filename: [xml_path1, xml_path2, ...] }
    """
    jobs_by_model = {}

    if not os.path.exists(pending_dir):
        return jobs_by_model

    for job_id in os.listdir(pending_dir):
        job_path = os.path.join(pending_dir, job_id)
        if not os.path.isdir(job_path):
            continue

        xml_path = os.path.join(job_path, "parameters.xml")
        if not os.path.exists(xml_path):
            continue

        try:
            params_client, param_model = parse_job_parameters(xml_path)

            # For UWS-compliant XMLs
            model_file = (
                param_model.get("ModelName")
                or param_model.get("ModelId")
                or param_model.get("CheckpointPath")
                #or params_client.get("filename")  # fallback for legacy format
            )

            if not model_file:
                logger.info(f"[SCHEDULER] Warning: model filename not found in {xml_path}")
                continue

            jobs_by_model.setdefault(model_file, []).append(xml_path)
        except Exception as e:
            logger.info(f"[SCHEDULER] Failed to parse {xml_path}: {e}")
            continue

    return jobs_by_model


def find_next_job_to_run():
    """
    Select the job with the lowest priority in WAITING.
    
    Returns:
        str or None: The process_id to run next, or None if none found.
    """
    log = load_job_log()
    waiting_jobs = {
        pid: entry for pid, entry in log.items()
        if entry["status"] == "WAITING" and os.path.isdir(os.path.join(JOB_PENDING, pid))
    }
    if not waiting_jobs:
        return None

    # Select job with lowest priority
    next_job = min(waiting_jobs.items(), key=lambda item: item[1]["priority"])
    return next_job[0]

def move_job_to_on_going(process_id):
    """
    Move a job from WAITING to ON_GOING.
    
    Args:
        process_id (str): Job identifier.
    
    Returns:
        str: Path to the new job folder in ON_GOING.
    """
    src = os.path.join(JOB_PENDING, process_id)
    dst = os.path.join(JOB_EXECUTING, process_id)
    shutil.move(src, dst)
    return dst


def scheduler_loop(poll_interval=2.0, max_wait_time=10.0):
    """
    Main loop of the job scheduler.
    Args:
        poll_interval (float): Time in seconds between polling for new jobs.
        max_wait_time (float): Maximum wait time for model loading.

    This loop continuously checks for new jobs in the PENDING directory,
    groups them by model, and executes them when the model is ready.
    """
    buffer = ModelJobBuffer(max_wait_time=max_wait_time)

    while True:
        try:
            # List of file in pending jobs directory
            pending_jobs = os.listdir(cfg.jobs_pending)

            # Add jobs to buffer
            for job_id in pending_jobs:
                job_path = os.path.join(cfg.jobs_pending, job_id)

                if not os.path.isdir(job_path):
                    continue

                xml_path = os.path.join(job_path, "parameters.xml")
                fits_path = os.path.join(job_path, "image.fits")

                try:
                    param_client, param_model = parse_job_parameters(xml_path)
                    model_filename = (
                        param_model.get("ModelName")
                        or param_model.get("ModelId")
                        or os.path.basename(param_model.get("CheckpointPath") or "")
                    )
                    if not model_filename:
                        raise ValueError(f"No model identifier found in {xml_path}")
                    buffer.add_job(job_id, model_filename)
                    logger.info(f"[SCHEDULER] Job {job_id} for model {model_filename} added to buffer.")
                except Exception as e:
                    logger.error(f"[SCHEDULER] Failed to parse job {job_id}: {e}")
                    move_to_error(job_id)
                    continue

            # Wait before next poll
            time.sleep(poll_interval)

        except Exception as e:
            logger.error(f"[SCHEDULER] Unexpected error: {e}")
            time.sleep(poll_interval)

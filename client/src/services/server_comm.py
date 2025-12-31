import time
import requests
import os
import logging

logger = logging.getLogger(__name__)


def poll_for_completion(server_url, job_id, user_id, poll_interval=10, timeout_s=600):
    """
    Periodically check if the UWS job is completed.

    Returns:
        str: terminal phase among {"COMPLETED","ERROR","CANCELLED","ABORTED","TIMEOUT"}.
    """
    t0 = time.time()
    while True:
        if (time.time() - t0) > timeout_s:
            logger.error(f"[CLIENT][poll_for_completion] TIMEOUT for Job ID: {job_id}[USER: {user_id}]")
            return "TIMEOUT"

        try:
            response = requests.get(f"{server_url}/jobs/{job_id}", timeout=10)
        except Exception as e:
            logger.error(f"[CLIENT][poll_for_completion] Request failed for Job {job_id}: {e}")
            time.sleep(poll_interval)
            continue

        if response.status_code == 200:
            phase = (response.json() or {}).get("phase")
            logger.info(f"[CLIENT][poll_for_completion] PHASE : {phase} for Job ID: {job_id}[USER: {user_id}]")

            if phase in ("COMPLETED", "ERROR", "CANCELLED", "ABORTED"):
                return phase
        else:
            logger.warning(f"[CLIENT][poll_for_completion] HTTP {response.status_code} for Job {job_id}")

        time.sleep(poll_interval)


def download_result(server_url, job_id, destination_folder):
    """
    Download result file from server after job completion.

    Args:
        server_url (str): Server base URL.
        job_id (str): Job ID to retrieve.
        destination_folder (str): Local folder to save the result.

    Returns:
        str: Path to the saved result file, or None if failed.
    """
    response = requests.get(f"{server_url}/jobs/{job_id}/results", stream=True)
    if response.status_code == 200:
        os.makedirs(destination_folder, exist_ok=True)
        file_path = os.path.join(destination_folder, f"net0_rts_{job_id}.dat")
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"[CLIENT] Result saved at {file_path}")
        return file_path
    else:
        logger.error("[CLIENT] Download error:", response.text)
        return None

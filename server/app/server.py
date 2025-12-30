from __future__ import annotations

import os, sys
import uuid
import time
from threading import Thread, Lock
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
from flask import (Flask, request, jsonify, send_file, Response)
from pathlib import Path

from queue import Queue, Full
from typing import Optional, Tuple, Deque, Dict, List
from collections import deque
from dataclasses import dataclass

# Importing custom modules
from utils.config import ServerConfig
from fileio.verify_fits import serve_models
from utils.job_logger import JOB_STORE as JOB_STORE
from core.pipeline import run_batch_and_finalize

import logging

logging.basicConfig(
    level=logging.INFO,  # passe à DEBUG si tu veux
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Flask App ---
app = Flask(__name__, static_url_path='/model-files', static_folder=None)
app.add_url_rule('/model-files/<path:filename>', 'serve_models', serve_models)

# Class JOB_LITE pour le batching des jobs
# 13/11/2025, c'est une refonte de la version actuelle qui est pourrie
@dataclass(frozen=True)
class JobLite:
    """Minimal job descriptor used for batching (no refonte du pipeline)."""
    job_id: str
    xml_path: Path
    fits_path: Path
    # seront complétés au draining (lecture XML)
    model_id: Optional[str] = None
    quantization: Optional[str] = None
    t_arrival: float = 0.0  # timestamp d’arrivée (pour SLA)

Key = Tuple[str, str]  # (model_id, quantization)


# --- Config file reading ---
cfg = ServerConfig()

PORT = cfg.get("server", "port", default=5000)

# Verbosity level for logging
VERBOSE = cfg.get("server", "verbose", default=True)

# Queue of jobs
QUEUE_MAXSIZE = cfg.get("policy", "max_jobs", default=10)
JOB_QUEUE: Queue[tuple[str, str, str]] = Queue(maxsize=QUEUE_MAXSIZE)
MAX_WORKERS = cfg.get("policy", "max_workers", default=2)

# BATCHING PARAMETERS
MIN_JOBS   = cfg.get("policy", "min_jobs", default=2)
MAX_BATCH   = cfg.get("policy", "max_batch", default=8)
MAX_WAIT_S = cfg.get("policy", "max_wait_s", default=2.0)
ROUND_ROBIN = cfg.get("policy", "round_robin", default=True)

# Cancellation tracking
CANCELLED_IDS: set[str] = set()
CANCEL_LOCK = Lock()

# Scheduler config
POLL_INTERVAL = cfg.get("scheduler", "poll_interval", default=2.0)
MAX_WAIT_TIME = cfg.get("scheduler", "max_wait_time", default=10.0)

# Directories for job states
QUEUED_DIR = cfg.jobs_queued
PENDING_DIR = cfg.jobs_pending
EXECUTING_DIR = cfg.jobs_executing
COMPLETED_DIR = cfg.jobs_completed
ERROR_DIR = cfg.jobs_error
CANCELLED_DIR = cfg.jobs_cancelled

for d in [PENDING_DIR, EXECUTING_DIR, COMPLETED_DIR, ERROR_DIR,
          CANCELLED_DIR, QUEUED_DIR]:
    os.makedirs(d, exist_ok=True)

MAX_UPLOAD_MB    = 1000       # Maximum upload size in megabytes
ALLOWED_XML_EXT  = {".xml"}   # to be sure that only XML files are uploaded
ALLOWED_FITS_EXT = {".fits"}  # to be sure that only FITS files are uploaded


sys.path.insert(0, cfg.cianna["path"])
import CIANNA as cnn

# Few helper functions
def _ext_ok(filename: str, allowed: set[str]) -> bool:
    return Path(filename).suffix.lower() in allowed


def _abort_json(status: int, msg: str):
    logger.warning("POST /jobs/ -> %s: %s", status, msg)
    return jsonify({"error": msg}), status


def _check_content_length(max_mb: int) -> bool:
    # Retourne True si la taille totale de requête est conforme
    cl = request.content_length
    if cl is None:
        return True  # pas d’info fournie par le client
    return cl <= max_mb * 1024 * 1024


# --- Buffers / état global batching --------------------------------
BUFFERS: Dict[Key, Deque[JobLite]] = {}
OLDEST_TS: Dict[Key, float] = {}     # ancienneté du 1er job par panier
KEYS: List[Key] = []                 # ordre des paniers (pour round-robin)
RR_INDEX = 0                         # index round-robin courant
BUFF_LOCK = Lock()                   # protège BUFFERS/OLDEST_TS/KEYS


def _extract_model_quant(xml_path: Path) -> Tuple[str, str]:
    """
    Extrait (model_id, quantization) depuis le XML côté client.
    Fallback sûrs si manquants.
    """
    try:
        from fileio.process_xml import extract_model_and_quant
        model, quant = extract_model_and_quant(str(xml_path))
        return (model or "default", quant or "FP32C_FP32A")
    except Exception:
        # fallback conservateur (si le XML est minimal)
        return ("default", "FP32C_FP32A")


def _enqueue_buffer(j: JobLite) -> None:
    """Range un job dans le bon panier (model, quant)."""
    global KEYS
    model, quant = j.model_id or "default", j.quantization or "FP32C_FP32A"
    key: Key = (model, quant)
    if key not in BUFFERS:
        BUFFERS[key] = deque()
        OLDEST_TS[key] = j.t_arrival
        KEYS.append(key)
    BUFFERS[key].append(j)
    # met à jour l'ancienneté du premier si nécessaire
    if len(BUFFERS[key]) == 1:
        OLDEST_TS[key] = j.t_arrival


def _drain_incoming_to_buffers() -> None:
    """
    Dépile rapidement la file d’entrée (JOB_QUEUE) vers les paniers
    en complétant model/quant via le XML. Pas de thread dédié: on draine
    au moment où un worker cherche un lot.
    """
    drained = 0
    while True:
        try:
            job_id, xml_p, fits_p = JOB_QUEUE.get_nowait()
        except Exception:
            break  # file vide

        with CANCEL_LOCK:
            if job_id in CANCELLED_IDS:
                # Marque le job comme CANCELLED et range le dossier
                try:
                    JOB_STORE.move_to(job_id, "CANCELLED")
                    JOB_STORE.set_phase(job_id, "CANCELLED", comment="Cancelled by client (PHASE=ABORT)")
                except Exception:
                    pass
                continue

        try:
            xml_path = Path(xml_p)
            fits_path = Path(fits_p)
            model, quant = _extract_model_quant(xml_path)
            j = JobLite(job_id=job_id, xml_path=xml_path, fits_path=fits_path,
                        model_id=model, quantization=quant, t_arrival=time.time())
            _enqueue_buffer(j)
        finally:
            JOB_QUEUE.task_done()
        drained += 1
    if drained:
        logger.debug("Drained %d jobs into buffers.", drained)


def _select_keys_round_robin() -> List[Key]:
    """Retourne la liste des clés dans l’ordre RR (pour l’équité)."""
    global RR_INDEX
    if not KEYS:
        return []
    if not ROUND_ROBIN:
        return list(KEYS)
    n = len(KEYS)
    if RR_INDEX >= n:
        RR_INDEX = 0
    order = KEYS[RR_INDEX:] + KEYS[:RR_INDEX]
    RR_INDEX = (RR_INDEX + 1) % n
    return order


def get_next_batch() -> Optional[Tuple[Key, List[JobLite], str]]:
    """
    Sélectionne un lot prêt selon les règles fixées.
    Retourne: ((model,quant), [jobs], reason) ou None si rien de prêt.
    """
    with BUFF_LOCK:
        _drain_incoming_to_buffers()  # intègre les nouveaux jobs

        now = time.time()
        # supprime les batchs vides
        for k in list(KEYS):
            if not BUFFERS.get(k):
                BUFFERS.pop(k, None)
                OLDEST_TS.pop(k, None)
                KEYS.remove(k)

        # rien à faire
        if not KEYS:
            return None

        # 1) priorité max-batch
        for k in _select_keys_round_robin():
            buf = BUFFERS.get(k)
            if not buf:
                continue
            if len(buf) >= MAX_BATCH:
                out = [buf.popleft() for _ in range(min(len(buf), MAX_BATCH))]
                # màj oldest
                OLDEST_TS[k] = out[0].t_arrival if buf else 0.0
                return (k, out, "max-batch")

        # 2) SLA: max_wait_s + min_jobs
        for k in _select_keys_round_robin():
            buf = BUFFERS.get(k)
            if not buf:
                continue
            oldest = OLDEST_TS.get(k, 0.0)
            if len(buf) >= MIN_JOBS and (now - oldest) >= MAX_WAIT_S:
                take = min(len(buf), MAX_BATCH)
                out = [buf.popleft() for _ in range(take)]
                OLDEST_TS[k] = out[0].t_arrival if buf else 0.0
                return (k, out, "sla-wait")

        # aucun lot prêt
        return None


def _job_worker(idx: int):
    """
    Worker de traitement
    - récupère un lot (même modèle/quant),
    - construit les entrées de jobs,
    - charge le modèle une fois,
    - exécute batch_prediction(...).
    """
    from core.pipeline import (
        make_job_entry,
        load_cianna_model_per_batch,
        get_model_info,
        XML_MODEL_PATH,
    )

    logger.info("Job worker #%d started", idx)
    idle_snooze = 0.05

    while True:
        try:
            # Récupère le batch prêt suivant
            batch_info = get_next_batch()  # ( (model, quant), [JobLite], reason ) ou None
            if batch_info is None:
                time.sleep(idle_snooze)
                continue

            (model_name, quant), jobs, reason = batch_info
            logger.info(
                "Worker #%d: flush model=%s quant=%s reason=%s size=%d",
                idx, model_name, quant, reason, len(jobs)
            )

            # 1) Construire les entries batch-ready
            entries = []
            model_path_ref = None
            first_params_model = None

            for j in jobs:
                try:
                    mp, entry = make_job_entry(j.job_id, str(j.xml_path), str(j.fits_path))
                    if model_path_ref is None:
                        model_path_ref = mp
                        first_params_model = entry.get("params_model") or {}
                    entries.append(entry)
                except Exception:
                    logger.exception("Worker #%d: job %s prepare failed", idx, j.job_id)
                    # Marque l’erreur proprement (facultatif si déjà géré côté pipeline)
                    try:
                        JOB_STORE.move_to_error(j.job_id, comment="prepare failed")
                    except Exception:
                        pass

            if not entries or model_path_ref is None:
                logger.warning("Worker #%d: empty batch after prepare", idx)
                continue

            # 2) Charger le modèle une fois pour le lot
            load_cianna_model_per_batch(model_path_ref, first_params_model or {})

            # 3) Base params depuis le registry (ĉ dans monitor_batch_buffers)
            try:
                name = first_params_model.get("ModelName") or \
                                first_params_model.get("ModelId") or model_name
                reg = get_model_info(XML_MODEL_PATH, name) or {}
            except Exception:
                reg = {}
            base_params = {**reg, **(first_params_model or {})}

            # 4) Exécuter le forward de TOUT le lot
            run_batch_and_finalize(entries, model_path_ref, base_params, cnn)

            logger.info("Worker #%d: batch done (model=%s quant=%s)",
                        idx, model_name, quant)

        except Exception:
            logger.exception("Worker #%d: unexpected error", idx)


def start_workers(n: int = MAX_WORKERS):
    """Démarre n workers de traitement des jobs.
    Un worker, c'est une boucle infinie qui dépile des lots et les exécute.
    
    """
    for i in range(n):
        t = Thread(target=_job_worker, args=(i+1,),
                   daemon=True, name=f"job-worker-{i+1}")
        t.start()
    logger.info("Started %d job workers (queue maxsize=%d)", n, QUEUE_MAXSIZE)


# Useful function for the rest of the code here
#######
def get_job_directory(job_id):
    """Retrieve the directory and current phase for a given job ID.

    Args:
        job_id (str): Unique identifier of the job.

    Returns:
        tuple: (job directory path, current phase name)
                or (None, None) if not found.
    """
    for state_dir in [PENDING_DIR, EXECUTING_DIR, COMPLETED_DIR,
                      ERROR_DIR, CANCELLED_DIR, QUEUED_DIR]:
        job_path = os.path.join(state_dir, job_id)
        if os.path.isdir(job_path):
            return job_path, os.path.basename(state_dir)
    return None, None


# Endpoint to create a new job / UWS compliant
#######
@app.route('/jobs/', methods=['POST'])
def submit_job():
    """
    Create a new asynchronous job by uploading XML and FITS files.
    - Validates presence + extensions
    - Saves under PENDING/<job_id> with secure filenames
    - Enqueues into JOB_QUEUE (workers will process)
    - Returns 303 See Other with Location: /jobs/<id>
    """

    # 0) Limite de taille globale des envois
    # Pour pas se faire peter le serveur avec des fichiers énormes
    if not _check_content_length(MAX_UPLOAD_MB):
        return _abort_json(413, f"Image too large (> {MAX_UPLOAD_MB} MB).")

    # 1) Récupération sécurisée
    xml_storage = request.files.get('xml')
    fits_storage = request.files.get('fits')
    if xml_storage is None or fits_storage is None:
        return jsonify({'error': 'Missing XML or FITS file.'}), 400

    # 2) Validation noms + extensions AVANT écriture
    if not xml_storage.filename or not fits_storage.filename:
        return _abort_json(400, "Empty filename for XML or FITS.")

    if not _ext_ok(xml_storage.filename, ALLOWED_XML_EXT):
        return _abort_json(400, "XML file must have .xml extension.")
    if not _ext_ok(fits_storage.filename, ALLOWED_FITS_EXT):
        return _abort_json(400, "FITS file must have .fits/.fit/.fts extension.")

    # 3) IDs, répertoires, chemins (Path)
    # Les dossiers portent le job_id dans le nom pour éviter les collisions
    process_id = str(uuid.uuid4())
    process_dir = Path(PENDING_DIR) / process_id
    process_dir.mkdir(parents=True, exist_ok=True)

    xml_name = secure_filename(xml_storage.filename)
    fits_name = secure_filename(fits_storage.filename)
    xml_path = process_dir / xml_name
    fits_path = process_dir / fits_name

    # 4) Sauvegarde fichiers
    try:
        xml_storage.save(str(xml_path))
        fits_storage.save(str(fits_path))
    except Exception as e:
        logger.exception("File save failed for job %s", process_id)
        try:
            xml_path.unlink(missing_ok=True)   # cleanup partiel du xml
            fits_path.unlink(missing_ok=True)  # cleanup partiel du fits
            if process_dir.exists() and not any(process_dir.iterdir()):
                process_dir.rmdir()
        except Exception:
            logger.warning("Cleanup failed for job %s", process_id)
        return _abort_json(500, f"File write failed: {e}")

    # 5) Log + enregistrement
    reception_date = datetime.now(timezone.utc).isoformat(timespec="seconds")
    JOB_STORE.create(process_id, fits_name)
    JOB_STORE.set_phase(process_id, "PENDING")
    logger.info(
        "Job %s received at %s (xml=%s, fits=%s)",
        process_id, reception_date, xml_name, fits_name
    )

    # 6) Enqueue
    try:
        JOB_QUEUE.put_nowait((process_id, str(xml_path), str(fits_path)))
        logger.info("Job %s enqueued (qsize=%d)", process_id, JOB_QUEUE.qsize())
    except Full:
        logger.warning("Queue full: rejecting job %s", process_id)
        try:
            xml_path.unlink(missing_ok=True)
            fits_path.unlink(missing_ok=True)
            if process_dir.exists() and not any(process_dir.iterdir()):
                process_dir.rmdir()
        except Exception:
            logger.exception("Cleanup failed for rejected job %s", process_id)
        return _abort_json(503, "Server busy, please retry later.")

    # 7) UWS: 303 See Other
    response = Response(status=303)
    response.headers['Location'] = f"/jobs/{process_id}"
    return response


#
# Endpoint to retrieve the output file (.dat) for a completed job
######
@app.route('/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    """Retrieve the output file (.dat) for a completed job.

    Args:
        job_id (str): Unique identifier of the job.

    Returns:
        Response: File download or error message.
    """
    job_dir, state_folder = get_job_directory(job_id)
    if job_dir is None:
        return jsonify({'error': 'Job not found.'}), 404

    if state_folder != 'COMPLETED':
        return jsonify({'error': f'Job not completed. Current state: {state_folder}'}), 400

    fwd_res_dir = os.path.join(job_dir, 'fwd_res')
    expected_file = f"net0_rts_{job_id}.dat"
    result_path = os.path.join(fwd_res_dir, expected_file)

    if not os.path.isfile(result_path):
        return jsonify({'error': 'Result file not found.'}), 404

    return send_file(result_path, as_attachment=True)


#
# Endpoint to retrieve the status and metadata of a specific job
######
def determine_phase(state_folder_name):
    """Map a directory name to its corresponding UWS phase.

    Args:
        state_folder_name (str): Name of the folder representing job phase.

    Returns:
        str: UWS phase corresponding to the folder name.
    """
    mapping = {
        'PENDING': 'PENDING',
        'EXECUTING': 'EXECUTING',
        'COMPLETED': 'COMPLETED',
        'ERROR': 'ERROR',
        'CANCELLED': 'CANCELLED',
        'QUEUED': 'QUEUED',
    }
    return mapping.get(state_folder_name, 'UNKNOWN')


#
# Helper function to generate XML response
######
def generate_xml_response(content_dict):
    """Generate a UWS-compliant XML response from a dictionary.

    Args:
        content_dict (dict): Dictionary containing job metadata.

    Returns:
        str: XML formatted string representing the job information.
    """
    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<uws:job xmlns:uws="http://www.ivoa.net/xml/UWS/v1.0">')
    for key, value in content_dict.items():
        xml.append(f'  <uws:{key}>{value}</uws:{key}>')
    xml.append('</uws:job>')
    return '\n'.join(xml)


#
# Endpoint to retrieve the status and metadata of a specific job
######
@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Retrieve the status and metadata of a specific job.

    Args:
        job_id (str): Unique identifier of the job.

    Returns:
        Response: Job metadata in JSON or XML format.
    """
    job_dir, state_folder = get_job_directory(job_id)
    if job_dir is None:
        return jsonify({'error': 'Job not found.'}), 404

    phase = determine_phase(state_folder)

    job_info = {
        'jobId': job_id,
        'phase': phase,
        'timestamp': datetime.now(timezone.utc).isoformat(timespec="seconds")
    }

    accept = request.headers.get('Accept', '')
    if 'application/xml' in accept:
        return Response(generate_xml_response(job_info),
                        mimetype='application/xml')
    else:
        return jsonify(job_info)
# ---------------------------------------------------------------------
# LA OU LA MAGIE OPERE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        HOST = cfg.get("server", "host", default="0.0.0.0")  # ex: "0.0.0.0"
    except Exception:
        HOST = "0.0.0.0"

    try:
        PORT = cfg.get("server", "port", default=5000)
    except Exception:
        PORT = 5000

    try:
        try:
            MAX_WORKERS = int(cfg.get("scheduler", "max_workers", default=MAX_WORKERS))
        except Exception:
            pass

        try:
            QUEUE_MAXSIZE = int(cfg.get("scheduler", "queue_maxsize", default=QUEUE_MAXSIZE))
        except Exception:
            pass

        start_workers(n=MAX_WORKERS)
    except Exception:
        logger.exception("Impossible to start job workers")
        raise

    logger.info("Flask server starting on %s:%s", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)

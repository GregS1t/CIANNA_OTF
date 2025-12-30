import os, json
import shutil
import threading
from datetime import datetime
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from contextlib import contextmanager
from xml.etree import ElementTree as ET


# 
import numpy as np

# Astropy imports
from astropy.io import fits
from astropy.wcs import WCS

# Custom imports
from utils.config import ServerConfig
from utils.data_gen import *
from utils.job_logger import update_job_status
from fileio.process_xml import get_model_info
from fileio.process_xml import parse_job_parameters
from utils.aux_fct import tile_filter, first_NMS


# Declare CIANNA path configuration and CIANNA import
cfg = ServerConfig("params/server.toml")
sys.path.insert(0, cfg.cianna["path"])
import CIANNA as cnn

###########################################################################
# POST PROCESSING PARAMETERS -> A mettre dans le fichier de config plus tard

first_nms_thresholds = np.array([0.01,-0.2,-0.4,-0.6])-.05 #lower is stricter
first_nms_obj_thresholds = np.array([1.0,0.7,0.5,0.35])

obj_thresholds = np.array([1.0,0.8,0.6,0.4])
second_nms_threshold = -0.15

val_med_lims = np.array([0.6,0.3,0.1])
val_med_obj  = np.array([0.8,0.6,0.4])

# Obj thresholds should be optimize for specif training or iter
# (using post_process.py)
# From CIANNA/examples/SKAO_SDC1/sdc1_pred_notebook.ipynd
# For best score (YOLO-CIANNA-ref)
prob_obj_cases = np.array([0.3101, 0.2759, 0.1536, 0.3101, 0.2759,
                           0.2314, 0.1146, 0.0962, 0.0468])

# For good precision (YOLO-CIANNA-alt)
# prob_obj_cases = np.array([0.678, 0.6271, 0.4915, 0.678, 0.6441,
#                            0.6102, 0.7288, 0.7458, 0.8305])

###########################################################################



# # Attention , problème avec la variable opt_thresholds, elle est définie dans le fichier
# # completed_v1.txt, mais n'est pas importée ici.
# #with open("./completed_v1.txt",'r') as file:
# #        opt_thresholds = np.array(file.readlines()[1].split(), dtype=np.float32)
# #
# prob_obj_cases = opt_thresholds + 0.4
# prob_obj_edges = prob_obj_cases + 0.0


# --------------------------------------------------------------------------- #
# Config & globals
# --------------------------------------------------------------------------- #

JOBS_PENDING    = cfg.jobs_pending
JOBS_EXECUTING  = cfg.jobs_executing
JOBS_COMPLETED  = cfg.jobs_completed
JOBS_ERROR      = cfg.jobs_error
MODEL_DIR       = cfg.models_dir
XML_MODEL_PATH  = cfg.model_registry_path
MAX_WAIT_TIME   = cfg.max_wait_time
BATCH_JOB_THRES = cfg.batch_jobs_thres
# Global CIANNA lock to prevent concurrent access
CIANNA_LOCK = threading.Lock()
ACTIVE_MODEL = {"path": None, "loaded": False}

# Queue per model
_QUEUES: Dict[str, Dict[str, Any]] = {}  # model_path -> {"jobs": [], "first_ts": float|None, "in_progress": bool}
_QUEUES_LOCK = threading.Lock()
_MODEL_LOCKS = defaultdict(threading.Lock)  # serialize batch_prediction per model



# --------------------------------------------------------------------------- #
# Utilities -> A sortir dans une fichier utils.py plus tard
# --------------------------------------------------------------------------- #

def as_int(x, default=None):
    """
    Coerce various scalar types/strings to int; tolerate '448x448' -> 448.
    """
    if isinstance(x, int):   return x
    if isinstance(x, float): return int(x)
    if isinstance(x, str):
        s = x.strip().strip('"').strip("'")
        if "x" in s or "X" in s:  # "448x448" -> "448"
            s = s.lower().split("x")[0]
        try:
            return int(s)
        except Exception:
            if default is not None:
                return default
            raise
    if default is not None:
        return default
    return int(x)

@contextmanager
def safe_chdir(path: str):
    """Temporarily change working directory and restore on exit."""
    prev = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def make_batch_tmp_dir() -> str:
    """Return a unique temporary batch directory under cfg.runtime_root."""
    base = getattr(cfg, "runtime_root", os.path.join(os.getcwd(), "runtime"))
    root = os.path.join(base, "batch_tmp")
    os.makedirs(root, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S_%f")
    return os.path.join(root, f"{stamp}_{uuid.uuid4().hex[:6]}")


def resolve_model_path(model_dir: str, model_name_or_id: str) -> str:
    """Return the on-disk path to the model file."""
    return os.path.join(model_dir, model_name_or_id)



# --------------------------------------------------------------------------- #
# Queue management (atomic)
# --------------------------------------------------------------------------- #

def enqueue_job(model_path: str, job: Dict[str, Any]) -> None:
    """Enqueue a job for a given model path; set the first timestamp if needed."""
    with _QUEUES_LOCK:
        q = _QUEUES.setdefault(model_path, {"jobs": [], "first_ts": None,
                                            "in_progress": False})
        q["jobs"].append(job)
        if q["first_ts"] is None:
            q["first_ts"] = time.time()


def try_snapshot_batch(model_path: str, max_size: int, max_wait: float) -> List[Dict[str, Any]] | None:
    """Atomically take the current job list if ready, set in_progress, and return it."""
    now = time.time()
    with _QUEUES_LOCK:
        q = _QUEUES.get(model_path)
        if not q or q["in_progress"]:
            return None
        size = len(q["jobs"])
        if size == 0:
            q["first_ts"] = None
            return None
        elapsed = now - (q["first_ts"] or now)
        if size >= max_size or elapsed >= max_wait:
            snapshot = q["jobs"]
            q["jobs"] = []
            q["first_ts"] = None
            q["in_progress"] = True
            return snapshot
        return None

def mark_batch_done(model_path: str) -> None:
    """Release in_progress flag when a batch ends."""
    with _QUEUES_LOCK:
        q = _QUEUES.get(model_path)
        if q:
            q["in_progress"] = False


def model_lock(model_path: str) -> threading.Lock:
    """Return the per-model lock used to serialize batch_prediction."""
    return _MODEL_LOCKS[model_path]


# --------------------------------------------------------------------------- #
# XML parsing (UWS parameters)
# --------------------------------------------------------------------------- #









def monitor_batch_buffers(poll_interval=1.0):
    """
    Background thread that monitors BATCH_BUFFER and launches prediction
    when MAX_WAIT_TIME is exceeded for a given model.

    - Regularly checks buffers grouped by model.
    - If the MAX_WAIT_TIME delay is exceeded, rereads parameters.xml
                                                        + CIANNA_models.xml
    - Merges parameters to obtain a complete dictionary.

    Args:
        poll_interval (float): Time in seconds between checks.

    """

    while True:
        try:
            now = time.time()

            for model_path, buffer in list(BATCH_BUFFER.items()):
                if model_path not in BUFFER_TIMESTAMPS or len(buffer) == 0:
                    continue
                
                elapsed = now - BUFFER_TIMESTAMPS[model_path] # Temps écoulé depuis le 1er job

                if elapsed >= MAX_WAIT_TIME:
                    print(f"[PIPELINE][MONITOR] Timeout reached for {model_path} "
                          f"(elapsed={elapsed:.1f}s, size={len(buffer)}). Launching batch...")

                    with CIANNA_LOCK:

                        # Read parameters from the first job in the buffer
                        try:
                            xml_path = os.path.join(buffer[0]["job_dir"], "parameters.xml")
                            if not os.path.exists(xml_path):
                                xml_candidates = [f for f in os.listdir(buffer[0]["job_dir"]) if f.endswith(".xml")]
                                if xml_candidates:
                                    xml_path = os.path.join(buffer[0]["job_dir"], xml_candidates[0])
                                else:
                                    raise FileNotFoundError("No XML found in job directory.")

                            #print(f"[MONITOR] Parsing XML: {xml_path}")
                            _, params_job = parse_job_parameters(xml_path)
                        except Exception as e:
                            print(f"[MONITOR] Warning: could not parse job XML for {model_path}: {e}")
                            params_job = buffer[0]["params_model"]

                        # Read the model info
                        try:
                            model_name = params_job.get("ModelName") or os.path.basename(model_path)
                            #model_name = os.path.splitext(model_name)[0]  # retire .dat
                            params_registry = get_model_info(XML_MODEL_PATH, model_name)
                            if params_registry:
                                print(f"[MONITOR] Found model info for {model_name} in registry.")
                            else:
                                print(f"[MONITOR] Warning: model {model_name} not found in registry.")
                                params_registry = {}
                        except Exception as e:
                            print(f"[MONITOR] Warning: could not load model info: {e}")
                            params_registry = {}

                        # Merge parameters from model registry and job
                        params_model = {**params_job, **params_registry}
                        #print(f"[MONITOR] Final merged parameter keys: {list(params_model.keys())[:8]} ...")

                        # Start batch prediction
                        try:
                            batch_prediction(buffer, model_path, params_model, cnn)
                        except Exception as e:
                            print(f"[MONITOR] Error during batch prediction for {model_path}: {e}")

                        # Clear buffer and timestamps
                        del BATCH_BUFFER[model_path]
                        BUFFER_TIMESTAMPS.pop(model_path, None)

                        for job in buffer:
                            pid = job["process_id"]
                            job_dir = job["job_dir"]
                            try:
                                update_job_status(
                                    pid,
                                    status="COMPLETED",
                                    comment="Batch completed (timeout)",
                                    end_time=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                                )
                                shutil.move(job_dir, os.path.join(cfg.jobs_completed, pid))
                                print(f"[PIPELINE][MONITOR] Job {pid} completed and moved")
                            except Exception as e:
                                print(f"[PIPELINE][MONITOR] Error moving job {pid} to COMPLETED: {e}")

            time.sleep(poll_interval)

        except Exception as e:
            print(f"[PIPELINE][MONITOR] Error in monitor thread: {e}")
            time.sleep(5.0)


def prepare_job_directory(base_output_dir, process_id):
    """
    Prepare the ON_GOING directory for a job and create subfolders if needed.

    Args:
        base_output_dir (str): Root output folder
        process_id (str): Unique identifier for the job.

    Returns:
        str: Path to the job directory.
    """
    job_dir = os.path.join(base_output_dir, process_id)
    os.makedirs(job_dir, exist_ok=True)

    fwd_res_dir = os.path.join(job_dir, "fwd_res")
    os.makedirs(fwd_res_dir, exist_ok=True)

    return job_dir


def validate_model_path(filename, model_dir):
    """
    Construct and verify the full path to the model.

    """

    model_path = os.path.join(model_dir, filename)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model root directory not found: {model_dir}")
    if not os.path.exists(os.path.dirname(model_path)):
        raise FileNotFoundError(f"Model directory not found: {os.path.dirname(model_path)}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model_path

def normalize_image(data, methods=["tanh"], scale=1.0, 
                    clip_min=None, clip_max=None):
    """
    Apply a sequence of normalization steps to a 2D image.

    Args:
        data (np.ndarray): 2D input image.
        methods (list of str): List of transformations to apply, in order.
                               Supported values: 'clip', 'linear', 'sqrt', 'log', 'tanh'.
        scale (float): Scaling factor used in applicable transformations.
        clip_min (float or None): Minimum value for clipping (used if 'clip' in methods).
        clip_max (float or None): Maximum value for clipping (used if 'clip' in methods).

    Returns:
        np.ndarray: Transformed image.

    Raises:
        ValueError: If an unknown method is provided.
    """
    import ast
    result = data.astype(np.float32)
    # print("[NORMALIZE] Started with methods:", methods)

    if isinstance(methods, str):
        try:
            parsed = ast.literal_eval(methods)
            if isinstance(parsed, (list, tuple)):
                methods = list(parsed)
            else:
                methods = [parsed]
        except (ValueError, SyntaxError):

            methods = [methods]


    if not isinstance(methods, (list, tuple)):
        methods = [methods]

    def _flatten(lst):
        """
            Subfunction to flatten a nested list or tuple.
        """
        flat = []
        for item in lst:
            if isinstance(item, (list, tuple)):
                flat.extend(_flatten(item))
            elif isinstance(item, str):
                txt = item.strip()
                if txt.startswith('[') and txt.endswith(']'):
                    try:
                        parsed = ast.literal_eval(txt)
                        flat.extend(_flatten(parsed))
                        continue
                    except (ValueError, SyntaxError):
                        pass
                flat.append(item)
            else:
                flat.append(item)
        return flat

    methods = _flatten(methods)

    for method in methods:
        method = method.strip().lower() # Just in case
        if method == "clip":
            if clip_min is not None and clip_max is not None:
                result = np.clip(result, clip_min, clip_max)
                result = (result - clip_min) / (clip_max - clip_min + 1e-12)
            else:
                result = (result - np.min(result)) / (np.max(result) - np.min(result) + 1e-12)

        elif method == "linear":
            result = scale * result

        elif method == "sqrt":
            result = np.sqrt(scale * result)

        elif method == "log":
            result = np.log1p(scale * result)

        elif method == "tanh":
            result = np.tanh(scale * result)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    # print("[NORMALIZE] Completed normalization.")
    # print("Shape of the data after normalization:", result.shape)

    return result


def normalize_and_patch(fits_path, params_client, params_model):
    """
    Normalize and tile a FITS image into input patches.

    Args:
        fits_path (str): Path to the FITS file.
        params_client (dict): Normalization parameters (e.g., method, min_pix, max_pix).
        params_model (dict): Model parameters (e.g., patch size, shift, offset).

    Returns:
        input_data (np.ndarray): 2D array of shape [n_patches, patch_size²].
        n_patches (int): Number of patches created from the image.
    """

    hdul = fits.open(fits_path)
    full_img = hdul[0].data[0, 0]
    hdul.close()

    # Normalize
    norm_fcts = params_client.get("norm_fct", "tanh")
    if isinstance(norm_fcts, str):
        norm_fcts = [norm_fcts]

    full_data_norm = normalize_image(
        full_img, methods=norm_fcts, scale=1.0,
        clip_min=float(params_client.get("min_pix", 0.4e-6)),
        clip_max=float(params_client.get("max_pix", 0.4e-4))
    )

    oid = params_model.get("OriginalInputDim") or params_model.get("InputSize") or 448

    image_size  = as_int(oid, 448)
    patch_shift = as_int(params_model.get("InferencePatchShift", 240))
    orig_offset = as_int(params_model.get("InferenceOrigOffset", 128))

    size_px, size_py = full_data_norm.shape
    nb_area_w = int((size_px - 128) / patch_shift) + 1
    nb_area_h = int((size_py - 128) / patch_shift) + 1
    n_patches = nb_area_w * nb_area_h

    input_data = np.zeros((n_patches, image_size * image_size), dtype="float32")
    patch = np.zeros((image_size, image_size), dtype="float32")

    for i_d in range(n_patches):
        
        p_y = int(i_d // nb_area_w)
        p_x = int(i_d % nb_area_w)

        xmin = p_x*patch_shift - orig_offset
        xmax = p_x*patch_shift + image_size - orig_offset
        ymin = p_y*patch_shift - orig_offset
        ymax = p_y*patch_shift + image_size - orig_offset

        px_min, px_max = 0, image_size
        py_min, py_max = 0, image_size

        set_zero = False

        if xmin < 0:
            px_min = -xmin
            xmin = 0
            set_zero = True
        if ymin < 0:
            py_min = -ymin
            ymin = 0
            set_zero = True
        if xmax > size_px:
            px_max = image_size - (xmax - size_px)
            xmax = size_px
            set_zero = True
        if ymax > size_py:
            py_max = image_size - (ymax - size_py)
            ymax = size_py
            set_zero = True

        if set_zero:
            patch[:, :] = 0.0

        patch[px_min:px_max, py_min:py_max] = np.flip(full_data_norm[xmin:xmax, ymin:ymax], axis=0)
        input_data[i_d, :] = patch.flatten("C")

    img_info = {
        "full_data_norm": full_data_norm,
        "nb_area_w": nb_area_w,
        "nb_area_h": nb_area_h,
        "patch_shift": patch_shift,
        "orig_offset": orig_offset,
        "image_size": image_size,
        "shape": full_data_norm.shape
    }

    return input_data, n_patches, img_info

#
# run_prediction_job modified taking a list of jobs to process in batch
#####
def run_prediction_job(process_id, xml_path, fits_path, model_dir=MODEL_DIR):
    """
    Main pipeline to run a prediction for a given job.
    Supports batch inference if multiple jobs use the same model.
    
    Args:
        process_id (str): Unique identifier for the job.
        xml_path (str): Path to the job's parameters XML file.
        fits_path (str): Path to the job's FITS file.
        model_dir (str): Directory where models are stored.
    
    """
    try:
        print(f"[run_prediction_job] Starting job: {process_id}")

        # --- Move job from PENDING to EXECUTING ---
        pending_job_dir = os.path.join(cfg.jobs_pending, process_id)
        executing_job_dir = os.path.join(cfg.jobs_executing, process_id)
        if os.path.exists(executing_job_dir):
            return
        if not os.path.exists(pending_job_dir):
            raise FileNotFoundError(f"[run_prediction_job] Job directory does not exist in PENDING: {pending_job_dir}")
        shutil.move(pending_job_dir, executing_job_dir)

        # --- Parse job parameters ---
        xml_path_exec = os.path.join(executing_job_dir, os.path.basename(xml_path))
        params_client, params_model = parse_job_parameters(xml_path_exec)

        # --- Extract FITS and model paths safely ---
        fits_path = (
            params_model.get("FITS_Path")      # UWS format
            or params_model.get("filename")    # legacy format
        )
        if not fits_path or not os.path.exists(fits_path):
            raise FileNotFoundError(f"[run_prediction_job] FITS file not found: {fits_path}")

        model_name = (
            params_model.get("ModelName")      # UWS
            or params_model.get("ModelId")
            or os.path.basename(params_model.get("CheckpointPath") or "")
        )
        if not model_name:
            raise ValueError("[run_prediction_job] Missing ModelName or CheckpointPath in parameters")

        job_model_path = validate_model_path(model_name, model_dir)

        # --- Prepare job entry ---
        job_entry = {
            "process_id": process_id,
            "job_dir": executing_job_dir,
            "fits_file": fits_path,
            "params_client": params_client,
            "params_model": params_model
        }

        # --- Model loading + batching ---
        with CIANNA_LOCK:
            import CIANNA as cnn

            # Load model if needed
            if ACTIVE_MODEL["path"] != job_model_path or not ACTIVE_MODEL["loaded"]:
                b_size = 8
                image_size = int(params_model.get("OriginalInputDim", "64x64x1x1").split("x")[0])
                cnn.init(
                    in_dim=i_ar([image_size, image_size]), in_nb_ch=1,
                    out_dim=0, bias=0.1, b_size=b_size, comp_meth='C_CUDA',
                    inference_only=1, dynamic_load=1,
                    mixed_precision=params_model.get("Quantization", "FP32C_FP32A"), adv_size=35
                )
                cnn.set_yolo_params()
                cnn.load(job_model_path, 0, bin=1)

                ACTIVE_MODEL["path"] = job_model_path
                ACTIVE_MODEL["loaded"] = True
            else:
                print(f"[run_prediction_job] Model already loaded: {job_model_path}")

            # --- Add job to buffer ---
            buffer = BATCH_BUFFER.setdefault(job_model_path, [])
            buffer.append(job_entry)

            # Register the timestamp for the first job of this model
            if job_model_path not in BUFFER_TIMESTAMPS:
                BUFFER_TIMESTAMPS[job_model_path] = time.time()

            # --- Check buffers for execution ---
            for mpath, buffer in list(BATCH_BUFFER.items()):
                elapsed = time.time() - BUFFER_TIMESTAMPS.get(mpath, time.time())
                print(f"[DEBUG] BUFFER_TIMESTAMPS keys: {list(BUFFER_TIMESTAMPS.keys())}")

                if len(buffer) >= BATCH_JOB_THRES or elapsed >= MAX_WAIT_TIME:

                    batch_prediction(buffer, mpath, params_model, cnn)
                    del BATCH_BUFFER[mpath]
                    BUFFER_TIMESTAMPS.pop(mpath, None)

                else:
                    print(f"[PIPELINE] Waiting for more jobs for model: {mpath} "
                          f"(current={len(buffer)}, elapsed={elapsed:.1f}s/{MAX_WAIT_TIME}s)")

    except Exception as e:
        update_job_status(process_id, status="ERROR", comment=str(e),
                          end_time=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        if os.path.exists(executing_job_dir):
            shutil.move(executing_job_dir, os.path.join(cfg.jobs_error, process_id))


def batch_prediction(job_list, model_path, model_params, cnn):
    """
    Run prediction on a batch of FITS files using a single model.
    Assumes model is already initialized and loaded.

    Args:
        job_list (list[dict]): List of job dictionaries (process_id, job_dir,
                        fits_file, etc.)
        model_path (str): Path to the loaded model file (.dat)
        model_params (dict): Full model parameters (merged from registry + job)
        cnn: CIANNA model interface
    """
    all_input_data = []
    all_patch_counts = []
    per_job_infos = []

    print(f"[BATCH] Processing {len(job_list)} jobs using model: {model_path}")

    for job in job_list:
        process_id = job["process_id"]
        job_dir = job["job_dir"]
        fits_file = job["fits_file"]
        params_client = job["params_client"]
        params_job = job["params_model"]

        # print(f"[BATCH] → Processing {fits_file}")
        # print("[BATCH] Using merged model parameters (subset):")
        # print({k: model_params[k] for k in list(model_params.keys())[:8]})

        # --- Merge des paramètres **par job** ---
        try:
            # Récup fiche registry pour ce modèle, si dispo
            model_name_j = params_job.get("ModelName") or params_job.get("ModelId")
            try:
                registry_params = get_model_info(XML_MODEL_PATH, model_name_j) if model_name_j else {}
            except Exception:
                registry_params = {}

            # Ordre de priorité: registry < model_params globaux < params du job
            merged_params = {**registry_params, **model_params, **params_job}
        except Exception as e:
            print(f"[BATCH] Warning: cannot merge params for {process_id}: {e}")
            merged_params = {**model_params, **params_job}

        try:
            # Use merged model parameters (not per-job ones)
            patch_array, n_patches, img_info = normalize_and_patch(fits_file,
                                                          params_client,
                                                          model_params)
            all_input_data.append(patch_array)
            all_patch_counts.append((process_id, patch_array.shape[0], job_dir))
            per_job_infos.append((process_id, job_dir, img_info, merged_params))
            #print(f"[BATCH] → Patch array shape: {patch_array.shape} ({n_patches} patches)")
        except Exception as e:
            print(f"[BATCH] Error on {process_id}: {e} during stacking and normalization")
            try:
                update_job_status(
                    process_id,
                    status="ERROR",
                    comment=str(e),
                    end_time=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                )
                shutil.move(job_dir, os.path.join(cfg.jobs_error, process_id))
            except Exception as e2:
                print(f"[BATCH] Error moving job {process_id} to ERROR: {e2}")

            continue

    if not all_input_data:
        raise RuntimeError("[BATCH] No valid inputs to process (all jobs failed pre-processing).")


    # --- Stack all inputs into a single numpy array ---
    full_input = np.vstack(all_input_data).astype(np.float32)
    total_patches = full_input.shape[0]
    #print(f"[BATCH] Full Shape: {full_input.shape}, Total patches: {total_patches}")
    outdim = 1
    targets = np.zeros((total_patches, outdim), dtype=np.float32)

    # --- Prepare a temporary runtime folder ---
    current_dir = os.getcwd()
    batch_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S_%f") + "_" + uuid.uuid4().hex[:6]
    batch_tmp_dir = os.path.join(current_dir, "runtime", "batch_tmp", batch_id)
    print(f"[BATCH] Using temporary directory: {batch_tmp_dir}")
    os.makedirs(batch_tmp_dir, exist_ok=True)
    os.chdir(batch_tmp_dir)

    # --- Run forward pass ---
    #print(f"[BATCH] Running forward on {total_patches} patches...")
    cnn.create_dataset("TEST", total_patches, full_input, targets)
    cnn.forward(repeat=1, no_error=1, saving=2, drop_mode="AVG_MODEL")

    # --- Read and dispatch predictions ---
    pred_file_path = os.path.join(batch_tmp_dir, "fwd_res", "net0_0000.dat")
    if not os.path.exists(pred_file_path):
        raise FileNotFoundError("[BATCH] Missing prediction file")

    predictions = np.fromfile(pred_file_path, dtype=np.float32)

    # Free CIANNA memory
    cnn.delete_dataset("TEST")

    print(f"[BATCH] Dispatching predictions to {len(all_patch_counts)} jobs...")
    offset = 0
    nb_box = int(params_job.get("YOLOBoxCount", 8))
    nb_param = int(params_job.get("YOLOParamCount", 4))
    c_size = int(params_job.get("YOLOGridCount"), 16)
    # VALEUR EN DUR A VIRER PLUS TARD
    fwd_image_size = 512
    yolo_nb_reg = int(fwd_image_size / c_size)


    for process_id, patch_count, job_dir in all_patch_counts:
        try:

            preds = predictions[offset: offset + patch_count]
            offset += patch_count

            fwd_dir = os.path.join(job_dir, "fwd_res")
            os.makedirs(fwd_dir, exist_ok=True)
            out_file = os.path.join(fwd_dir, f"net0_rts_{process_id}.dat")
            preds.astype(np.float32).tofile(out_file)
            print(f"[BATCH] Saved prediction for {process_id}")

            # POST-PROCESSING TO XML AND VOTABLE
            # Using pred2xml function
            try:
                pred2csv(out_file, img_info, prob_obj_cases,
                    val_med_lims, val_med_obj,
                    first_nms_thresholds, first_nms_obj_thresholds,
                    nb_box, nb_param, yolo_nb_reg)
                print(f"[BATCH] Post-processing completed for {process_id}")
                
            except Exception as e:
                print(f"[BATCH] Error during post-processing for {process_id}: {e}")


            # Update the job status.
            update_job_status(
                process_id,
                status="COMPLETED",
                comment="Batch completed",
                end_time=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            )
            shutil.move(job_dir, os.path.join(cfg.jobs_completed, process_id))
            print(f"[PIPELINE][BATCH] Job {process_id} completed and moved")

        except Exception as e:
            print(f"[BATCH] Error saving prediction for {process_id}: {e}")
            try:
                update_job_status(
                    process_id,
                    status="ERROR",
                    comment=str(e),
                    end_time=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                )
                shutil.move(job_dir, os.path.join(cfg.jobs_error, process_id))
            except Exception as e2:
                print(f"[BATCH] Error moving job {process_id} to ERROR: {e2}")

    # --- Cleanup temporary files ---
    try:
        shutil.rmtree(batch_tmp_dir, ignore_errors=True)
    except Exception as e:
        print(f"[BATCH] Warning during temp cleanup: {e}")


    os.chdir(current_dir)
    print("[BATCH] Batch completed and cleaned up.")


#####   INFERENCE RELATED GLOBAL VARIABLES    #####


### Function to postprocess the prediction into XML format
BATCH_JOB_THRES = 4  # <- à adapter si besoin
def pred2csv(dat_path, img_info, prob_obj_cases,
                val_med_lims, val_med_obj,
                first_nms_thresholds, first_nms_obj_thresholds,
                nb_box, nb_param, yolo_nb_reg):
    """
    Postprocesses a CIANNA .dat prediction file and converts it to an XML file.

    Applies YOLO-style filtering and non-maximum suppression (NMS) on the
    prediction tensor extracted from the binary `.dat` file and saves the
    resulting boxes into an XML format.
    """

    full_data_norm=img_info["full_data_norm"]
    nb_area_w=img_info["nb_area_w"]
    nb_area_h=img_info["nb_area_h"]
    patch_shift=img_info["patch_shift"]
    orig_offset=img_info["orig_offset"]
    fwd_image_size=img_info["image_size"]

    pred_data = np.fromfile(dat_path, dtype="float32")

    repeat = 1

    predict = np.reshape(pred_data, (repeat, nb_area_h, nb_area_w,
                                     nb_box * (8 + nb_param),
                                     yolo_nb_reg, yolo_nb_reg))
    predict = np.mean(predict, axis=0)

    print(np.shape(predict))
    print("1st order predictions filtering ...")

    final_boxes = []
    c_tile = np.zeros((yolo_nb_reg * yolo_nb_reg * nb_box,
                       (6 + 1 + nb_param + 1)), dtype="float32")
    c_tile_kept = np.zeros_like(c_tile)
    c_box = np.zeros((6 + 1 + nb_param + 1), dtype="float32")
    patch = np.zeros((fwd_image_size, fwd_image_size), dtype="float32")
    box_count_per_reg_hist = np.zeros((nb_box + 1), dtype="int")

    cumul_filter_box = 0
    cumul_post_nms = 0

    for ph in tqdm(range(nb_area_h)):
        for pw in range(nb_area_w):
            c_tile[:, :] = 0.0
            c_tile_kept[:, :] = 0.0

            xmin = pw * patch_shift - orig_offset
            xmax = xmin + fwd_image_size
            ymin = ph * patch_shift - orig_offset
            ymax = ymin + fwd_image_size

            if (ph == 0 or ph == nb_area_h - 1 or pw == 0 or pw == nb_area_w - 1):
                patch[:, :] = 0.0
            else:
                patch[:, :] = full_data_norm[ymin:ymax, xmin:xmax]

            c_pred = predict[ph, pw, :, :, :]
            c_nb_box = tile_filter(c_pred, c_box, c_tile, nb_box, prob_obj_cases,
                                   patch, val_med_lims, val_med_obj, box_count_per_reg_hist)

            cumul_filter_box += c_nb_box
            c_nb_box_final = first_NMS(c_tile, c_tile_kept, c_box, c_nb_box,
                                       first_nms_thresholds, first_nms_obj_thresholds)
            cumul_post_nms += c_nb_box_final

            if ph < 2 or ph >= nb_area_h - 2 or pw < 2 or pw >= nb_area_w - 2:
                c_nb_box_final = 0

            final_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))

    flat_kept = np.vstack(final_boxes)

    print("NMS removed average frac:", (cumul_filter_box - cumul_post_nms) / cumul_filter_box)
    

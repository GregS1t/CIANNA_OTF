import os, sys
import shutil
import threading
from datetime import datetime, timezone
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from contextlib import contextmanager

import numpy as np
import numpy as np
from tqdm import tqdm

# Astropy imports
from astropy.io import fits
from astropy.table import Table
from astropy.io.votable import from_table, writeto, parse

# Custom imports
from utils.config import ServerConfig
from utils.data_gen import *
from utils.job_logger import update_job_status
from utils.job_logger import (JOB_STORE as JOB_STORE,
                              move_to_completed, move_to_error)
from fileio.process_xml import get_model_info
from fileio.process_xml import parse_job_parameters
from utils.aux_fct import tile_filter, first_NMS

import logging
logger = logging.getLogger(__name__)

# Declare CIANNA path configuration and CIANNA import
cfg = ServerConfig("params/server.toml")
sys.path.insert(0, cfg.cianna["path"])
import CIANNA as cnn


###########################################################################
# POST PROCESSING PARAMETERS -> A mettre dans le fichier de config plus tard

# box_prior_class = np.array([0,0,0,0,0,1,1,4], dtype="int")

# first_nms_thresholds = np.array([0.01, -0.2, -0.4, -0.6]) - .05  #lower is stricter
# first_nms_obj_thresholds = np.array([1.0, 0.7, 0.5, 0.35])

# obj_thresholds = np.array([1.0, 0.8, 0.6, 0.4])
# second_nms_threshold = -0.15

# val_med_lims = np.array([0.6, 0.3, 0.1])
# val_med_obj  = np.array([0.8, 0.6, 0.4])

# Obj thresholds should be optimize for specif training or iter
# (using post_process.py)
# From CIANNA/examples/SKAO_SDC1/sdc1_pred_notebook.ipynd
# For best score (YOLO-CIANNA-ref)
# prob_obj_cases = np.array([0.3101, 0.2759, 0.1536, 0.3101, 0.2759,
#                            0.2314, 0.1146, 0.0962, 0.0468])

# For good precision (YOLO-CIANNA-alt)
# prob_obj_cases = np.array([0.678, 0.6271, 0.4915, 0.678, 0.6441,
#                            0.6102, 0.7288, 0.7458, 0.8305])

################################################################################


# ---------------------------------------------------------------------------- #
# Config & globals
# ---------------------------------------------------------------------------- #
JOBS_QUEUED = cfg.jobs_queued
JOBS_PENDING = cfg.jobs_pending
JOBS_EXECUTING = cfg.jobs_executing
JOBS_COMPLETED = cfg.jobs_completed
JOBS_ERROR = cfg.jobs_error
JOBS_ABORTED = cfg.jobs_aborted
MODEL_DIR = cfg.models_dir
XML_MODEL_PATH = cfg.model_registry_path
MAX_WAIT_TIME = cfg.max_wait_time
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
# Queue management
# --------------------------------------------------------------------------- #
def enqueue_job(model_path: str, job: Dict[str, Any]) -> None:
    """
    Enqueue a job for a given model path; set the first timestamp if needed.
    Duplicate process_id for the same model_path are ignored (idempotent).
    """
    pid = job.get("process_id")
    if not pid:
        return

    with _QUEUES_LOCK:
        q = _QUEUES.setdefault(model_path, {
            "jobs": [], "first_ts": None, "in_progress": False, "ids": set()
        })
        if pid in q["ids"]:
            # déjà dans la file pour ce modèle → on ignore
            logger.info(f"[QUEUE] skip duplicate {pid} for {model_path}")
            return

        q["jobs"].append(job)
        q["ids"].add(pid)
        if q["first_ts"] is None:
            q["first_ts"] = time.time()


def finalize_job_id(model_path: str, process_id: str) -> None:
    """
    Remove a process_id from the per-model 'ids' set after finalization.
    """
    with _QUEUES_LOCK:
        q = _QUEUES.get(model_path)
        if q and "ids" in q:
            q["ids"].discard(process_id)


def try_snapshot_batch(model_path: str, max_size: int, max_wait: float):
    """
    Atomically take the current job list if ready, set in_progress, 
    and return it.
    """
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
    # logger.info("[NORMALIZE] Started with methods:", methods)

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
    return result


def normalize_and_patch_per_img(fits_path, params_client, params_model):
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
    logger.info(f"[NORMALIZE AND PATCH] Loading FITS: {fits_path}")

    hdul = fits.open(fits_path)
    full_img = hdul[0].data[0, 0]
    hdul.close()

    # Normalize
    norm_fcts = params_client.get("norm_fct", "tanh")
    logger.info(f"[NORMALIZE AND PATCH] Normalization functions: {norm_fcts}")

    if isinstance(norm_fcts, str):
        norm_fcts = [norm_fcts]

    full_data_norm = normalize_image(
        full_img, methods=norm_fcts, scale=1.0,
        clip_min=float(params_client.get("min_pix", 0.4e-6)),
        clip_max=float(params_client.get("max_pix", 0.4e-4))
    )

    oid = params_model.get("OriginalInputDim") or params_model.get("InputSize")
    image_size = as_int(oid, 512)
    patch_shift = as_int(params_model.get("InferencePatchShift", 240))
    orig_offset = as_int(params_model.get("InferenceOrigOffset", 128))

    size_px, size_py = full_data_norm.shape
    nb_area_w = int((size_px - orig_offset) / patch_shift) + 1
    nb_area_h = int((size_py - orig_offset) / patch_shift) + 1
    n_patches = nb_area_w * nb_area_h

    patched_data = np.zeros((n_patches, image_size * image_size), dtype="float32")
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
        patched_data[i_d, :] = patch.flatten("C")

    img_info = {
        "full_data_norm": full_data_norm,
        "nb_area_w": nb_area_w,
        "nb_area_h": nb_area_h,
        "patch_shift": patch_shift,
        "orig_offset": orig_offset,
        "image_size": image_size,
        "shape": full_data_norm.shape
    }

    return patched_data, n_patches, img_info


def make_job_entry(process_id: str, xml_path: str, fits_path: str,
                   model_dir: str = MODEL_DIR) -> tuple[str, dict]:
    """
    Build a job entry (dict) compatible with batch_prediction(...)
    WITHOUT enqueuing it in internal queues.
    Returns (model_path, job_dict).
    """
    client, model = parse_job_parameters(xml_path)
    if fits_path:
        client["FITS_Path"] = fits_path

    model_name = model.get("ModelName") or model.get("ModelId")
    if not model_name:
        raise ValueError("ModelName/ModelId missing in XML")

    model_path = resolve_model_path(model_dir, model_name)

    job_dir = os.path.join(JOBS_EXECUTING, process_id)
    os.makedirs(job_dir, exist_ok=True)

    entry = {
        "process_id": process_id,
        "job_dir": job_dir,
        "fits_file": client.get("FITS_Path"),
        "params_client": client,
        "params_model": model,
    }
    return model_path, entry


def load_cianna_model_per_batch(model_path: str, model_params: dict) -> None:
    """
    Ensure the CIANNA network is initialized/loaded for this model_path once.
    Uses global CIANNA_LOCK / ACTIVE_MODEL to serialize the load/init.

    Args:
        model_path (str): Path to the CIANNA model file.
        model_params (dict): Model parameters from registry + XML.
    Raises:
        Various exceptions if model loading fails.
    Returns:
        None.
    """
    with CIANNA_LOCK:
        need_load = (ACTIVE_MODEL["path"] != model_path) or (not ACTIVE_MODEL["loaded"])
        if not need_load:
            logger.info("[load_cianna_model_per_batch] model ready: %s",
                        model_path)
            return

        # Minimal init
        in_size = as_int(model_params.get("OriginalInputDim", 256))
        quant = model_params.get("Quantization", "FP32C_FP32A")
        batch_size = as_int(model_params.get("MaxBatchSize", 16), 16)
        bias = float(model_params.get("BiasInit", 0.1))

       # Initialize and load model
        cnn.init(
            in_dim=i_ar([in_size, in_size]), in_nb_ch=1, out_dim=0,
            bias=bias, b_size=batch_size, comp_meth='C_CUDA', inference_only=1,
            dynamic_load=1, mixed_precision=quant, adv_size=35, no_logo=1)
        
        cnn.set_yolo_params()
        cnn.load(model_path, 0, bin=1)

        ACTIVE_MODEL.update({"path": model_path, "loaded": True})
        logger.info("[load_cianna_model_per_batch] model loaded: %s", model_path)


def run_batch_and_finalize(entries: list[dict], model_path: str,
                           base_params: dict, cnn,
                           is_cancelled=None) -> None:
    """
    Execute a batch of jobs for a given model_path.
    Update job phases/directories accordingly.
        - set all jobs to EXECUTING at start
        - if all goes well -> COMPLETED
        - on exception -> ERROR for remaining jobs
    Args:
        entries (list of dict): List of job entries for batch_prediction(...).
        model_path (str): Path to the CIANNA model file.
        base_params (dict): Base model parameters from registry + XML.
        cnn: The CIANNA module/object.
    Raises:
        Various exceptions if processing or file operations fail.
    Returns:
        None
    """
    # 0) Check if cancelled before starting
    if is_cancelled is None:
        def is_cancelled(_pid: str) -> bool:
            return False

    active_entries = []
    for e in entries:
        pid = e.get("process_id")
        if pid and is_cancelled(pid):
            try:
                JOB_STORE.move_to(pid, "ABORTED")
                JOB_STORE.set_phase(pid, "ABORTED", comment="Aborted before execution")
            except Exception:
                pass
        else:
            active_entries.append(e)

    # nothing to do
    if not active_entries:
        return



    # 1)  EXECUTING 
    for e in active_entries:
        pid = e.get("process_id")
        try:
            JOB_STORE.set_phase(pid, "EXECUTING")
        except Exception:
            logger.exception("[BATCH] set EXECUTING failed for %s", pid)

    # 2) Execute the forward pass on the batch
    try:
        batch_prediction(active_entries, model_path, base_params, cnn)
    except Exception as err:
        logger.exception("[BATCH] batch_prediction failed: %s", err)
        # 3a) tous en ERROR (ceux non déjà finalisés par batch_prediction)
        for e in active_entries:
            pid = e.get("process_id")
            try:
                move_to_error(pid, comment=str(err))
            except Exception:
                logger.exception("[BATCH] move_to_error failed %s", pid)
        # remonter pour log worker
        raise
    else:
        # 3b) tous en COMPLETED (si batch_prediction ne l’a pas déjà fait)
        for e in active_entries:
            pid = e.get("process_id")
            if pid and is_cancelled(pid):
                try:
                    JOB_STORE.move_to(pid, "ABORTED")
                    JOB_STORE.set_phase(pid, "ABORTED",
                                        comment="Aborted during execution")
                except Exception:
                    pass
                continue
            try:
                move_to_completed(pid)
            except Exception:
                logger.exception("[BATCH] move_to_completed failed %s", pid)

def _post_process_dat(dat_path, img_info, prob_obj_cases, val_med_lims,
                    val_med_obj, first_nms_thresholds, 
                    first_nms_obj_thresholds, nb_box, nb_param, yolo_nb_reg,
                    box_prior_class):
    """
    Read the CIANNA .dat file, apply YOLO + NMS filtering, and return flat_kept.
    
    Args:
        dat_path (str): Path to the CIANNA .dat file.
        img_info (dict): Image information from preprocessing.
        prob_obj_cases (np.ndarray): Probability thresholds for objectness.
        val_med_lims (np.ndarray): Median value limits for filtering.
        val_med_obj (np.ndarray): Median objectness values for filtering.
        first_nms_thresholds (np.ndarray): Thresholds for first NMS step.
        first_nms_obj_thresholds (np.ndarray): Objectness thresholds for first NMS.
        nb_box (int): Number of boxes per grid cell.
        nb_param (int): Number of additional parameters per box.
        yolo_nb_reg (int): 
    --------
    Returns:
    flat_kept : np.ndarray
        2D array of kept boxes after filtering and NMS.
        Each line : RA, DEC, X, Y, W, H, Objectness, Flux_Jy, BMAJ, BMIN, PA, ...
    """

    full_data_norm = img_info["full_data_norm"]
    nb_area_w      = img_info["nb_area_w"]
    nb_area_h      = img_info["nb_area_h"]
    patch_shift    = img_info["patch_shift"]
    orig_offset    = img_info["orig_offset"]
    fwd_image_size = img_info["image_size"]

    pred_data = np.fromfile(dat_path, dtype="float32")

    repeat = 1

    try:
        predict = np.reshape(
        pred_data,
            (repeat, nb_area_h, nb_area_w,
            nb_box * (8 + nb_param),
            yolo_nb_reg, yolo_nb_reg))

    except Exception as e:
        print("Error in reshaping .dat predictions:", e)
        return np.zeros((0, 0), dtype="float32")

    predict = np.mean(predict, axis=0)
    final_boxes = []

    c_tile = np.zeros(
        (yolo_nb_reg * yolo_nb_reg * nb_box, (6 + 1 + nb_param + 1)),
        dtype="float32"
    )
    c_tile_kept = np.zeros_like(c_tile)
    c_box = np.zeros((6 + 1 + nb_param + 1), dtype="float32")
    patch = np.zeros((fwd_image_size, fwd_image_size), dtype="float32")
    box_count_per_reg_hist = np.zeros((nb_box + 1), dtype="int")

    cumul_filter_box = 0
    cumul_post_nms   = 0
    for ph in tqdm(range(0, nb_area_h), desc="Tiles H"):
        for pw in range(0, nb_area_w):
            c_tile[:, :] = 0.0
            c_tile_kept[:, :] = 0.0

            xmin = pw * patch_shift - orig_offset
            xmax = xmin + fwd_image_size
            ymin = ph * patch_shift - orig_offset
            ymax = ymin + fwd_image_size

            if (ph == 0 or ph == nb_area_h - 1 or
                pw == 0 or pw == nb_area_w - 1):
                patch[:, :] = 0.0
            else:
                patch[:, :] = full_data_norm[ymin:ymax, xmin:xmax]

            c_pred = predict[ph, pw, :, :, :]

            c_nb_box = tile_filter(
                c_pred, c_box, c_tile,
                nb_box, prob_obj_cases, patch,
                val_med_lims, val_med_obj, box_count_per_reg_hist, yolo_nb_reg, 
                nb_param, fwd_image_size)
            
            cumul_filter_box += c_nb_box
            c_nb_box_final = first_NMS(
                c_tile, c_tile_kept, c_box, c_nb_box, box_prior_class,
                first_nms_thresholds, first_nms_obj_thresholds
            )
            cumul_post_nms += c_nb_box_final

            # bord de mosaïque : on coupe
            if ph < 2 or ph >= nb_area_h - 2 or pw < 2 or pw >= nb_area_w - 2:
                c_nb_box_final = 0

            if c_nb_box_final > 0:
                final_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))

    if cumul_filter_box > 0:
        print(
            "NMS removed average frac:",
            (cumul_filter_box - cumul_post_nms) / cumul_filter_box
        )

    if not final_boxes:
        print("No boxes kept after NMS.")
        return np.zeros((0, 0), dtype="float32")
    flat_kept = np.vstack(final_boxes)
    return flat_kept


def pred2votable(dat_path, img_info, prob_obj_cases, val_med_lims,
                    val_med_obj, box_prior_class,
                    first_nms_thresholds, first_nms_obj_thresholds,
                    nb_box, nb_param, yolo_nb_reg, votable_path):
    """
    Convert CIANNA .dat predictions to VOTable format.

    Args:
        dat_path (str): Path to the CIANNA .dat file.
        img_info (dict): Image information from preprocessing.
        prob_obj_cases (np.ndarray): Probability thresholds for objectness.
        val_med_lims (np.ndarray): Median value limits for filtering.
        val_med_obj (np.ndarray): Median objectness values for filtering.
        first_nms_thresholds (np.ndarray): Thresholds for first NMS step.
        first_nms_obj_thresholds (np.ndarray): Objectness thresholds for first NMS.
        nb_box (int): Number of boxes per grid cell.
        nb_param (int): Number of additional parameters per box.
        yolo_nb_reg (int): Number of regression outputs per box.
        votable_path (str): Output path for the VOTable file.

    Columns in VOTable:
      RA(deg), DEC(deg), X(pix), Y(pix), W(pix), H(pix),
      Objectness(real), Apparent Flux(Jy),
      BMAJ(arcsec), BMIN(arcsec), PA(degree), ...

    Returns:
        None
    
    Raises:
        ValueError: If the processed data has insufficient columns.
    """
    print("Before _post_process_dat")
    flat_kept = _post_process_dat(dat_path, img_info, prob_obj_cases,
                                    val_med_lims, val_med_obj, first_nms_thresholds,
                                    first_nms_obj_thresholds, nb_box, nb_param,
                                    yolo_nb_reg, box_prior_class)

    if flat_kept.size == 0:
        logger.error("pred2votable: no detections, writing empty VOTable.")
        tbl = Table()
        vot = from_table(tbl)
        writeto(vot, votable_path)
        return

    n_rows, n_cols = flat_kept.shape

    if n_cols < 11:
        raise ValueError(
            f"pred2votable: expected at least 11 columns, got {n_cols}"
        )

    tbl = Table()

    # Columns description
    tbl["RA"]        = flat_kept[:, 0]  # deg
    tbl["DEC"]       = flat_kept[:, 1]  # deg
    tbl["X_IMAGE"]   = flat_kept[:, 2]  # pix
    tbl["Y_IMAGE"]   = flat_kept[:, 3]  # pix
    tbl["WIDTH"]     = flat_kept[:, 4]  # pix
    tbl["HEIGHT"]    = flat_kept[:, 5]  # pix
    tbl["OBJECTNESS"] = flat_kept[:, 6]  # unitless
    tbl["FLUX_JY"]    = flat_kept[:, 7]  # Jy
    tbl["BMAJ"]       = flat_kept[:, 8]  # arcsec
    tbl["BMIN"]       = flat_kept[:, 9]  # arcsec
    tbl["PA"]         = flat_kept[:,10]  # degree

    if n_cols > 11:
        extra = n_cols - 11
        for i in range(extra):
            tbl[f"EXTRA{i}"] = flat_kept[:, 11 + i]

    tbl.meta["ORIGIN"] = "CIANNA-OTF"
    tbl.meta["NDET"]   = n_rows

    vot = from_table(tbl)
    writeto(vot, votable_path)
    logger.info(f"VOTable written: {votable_path} (rows={n_rows}, cols={n_cols})")


def preview_votable(vot_path: str, n: int = 5) -> None:
    """
    Read a VOTable file and print the first 'n' lines to the console.

    Parameters
    ----------
    vot_path : str
        Path to the VOTable file.
    n : int, optional
        Number of lines to preview (default is 5).

    Notes
    -----
    This function uses Astropy to parse and pretty-print the VOTable.
    """
    try:
        vot = parse(vot_path)
        table = vot.get_first_table().to_table()

        # Si le fichier est vide
        if len(table) == 0:
            print(f"[INFO] VOTable '{vot_path}' est vide.")
            return

        # Déterminer combien de lignes afficher
        n_preview = min(n, len(table))

        print(f"\n=== Aperçu du VOTable : {vot_path} ===")
        print(f"Nombre total de lignes : {len(table)}")
        print(f"Affichage des {n_preview} premières lignes :\n")

        # Astropy sait pretty-printer les tables
        print(table[:n_preview])

    except Exception as e:
        print(f"[ERREUR] Impossible de lire le VOTable '{vot_path}': {e}")


# ------------------------------------------------------------------------------
# Batch inference
# ------------------------------------------------------------------------------
def batch_prediction(jobs: List[Dict[str, Any]], model_path: str,
                     base_params: Dict[str, Any], cianna: Any) -> None:
    """
    Run one forward pass for all jobs bound to the same model; 
    dispatch outputs per job.

    Args:
        jobs (List[Dict[str, Any]]): List of job entries, each containing:
            - "process_id": str
            - "job_dir": str
            - "fits_file": str
            - "params_client": Dict[str, Any]
            - "params_model": Dict[str, Any]
        model_path (str): Path to the CIANNA model file.
        base_params (Dict[str, Any]): Base model parameters from registry + XML.
        cianna (Any): The CIANNA module/object.

    Raises:
        Various exceptions if processing or file operations fail.

    Returns:
        None
    """
    lock = model_lock(model_path) # serialize per-model batch_prediction
    with lock:
        logger.info(f"[BATCH] start: {len(jobs)} jobs, model={model_path}")

        patch_data_list: List[np.ndarray] = []
        counts: List[Tuple[str, int, str]] = []  # (pid, n_patches, job_dir)
        per_job_meta: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []

        for job in jobs:
            pid = job["process_id"]
            job_dir = job["job_dir"]
            fits_file = job["fits_file"]
            p_client = job.get("params_client", {})
            p_job = job.get("params_model", {})

            # Merge params (registry < base < job)
            name = p_job.get("ModelName") or p_job.get("ModelId")
            try:
                model_info = get_model_info(XML_MODEL_PATH, name) if name else {}
                if model_info:

                    postproc = {
                        "BoxPriorClass": ("box_prior_class", np.int32),
                        "FirstNMSThresholds": ("first_nms_thresholds", np.float32),
                        "FirstNMSObjThresholds": ("first_nms_obj_thresholds", np.float32),
                        "ObjThresholds": ("obj_thresholds", np.float32),
                        "ValMedLims": ("val_med_lims", np.float32),
                        "ValMedObj": ("val_med_obj", np.float32),
                        "ProbObjCases": ("prob_obj_cases", np.float32),
                    }
                    for key, (glob_name, dtype) in postproc.items():
                        val = model_info.get(key)
                        if val is None or val == "":
                            continue
                        if isinstance(val, np.ndarray):
                            arr = val.astype(dtype, copy=False)
                        elif isinstance(val, (list, tuple)):
                            arr = np.asarray(val, dtype=dtype)
                        else:
                            arr = np.fromstring(str(val), sep=",", dtype=dtype)
                        if arr.size:
                            globals()[glob_name] = arr
                            model_info[key] = arr
                    nms_val = model_info.get("SecondNMSThreshold")
                    if nms_val not in (None, ""):
                        try:
                            second_nms_threshold = float(nms_val)
                            globals()["second_nms_threshold"] = second_nms_threshold
                            model_info["SecondNMSThreshold"] = second_nms_threshold
                        except Exception:
                            pass

                    box_prior_class = model_info.get("BoxPriorClass")
                    first_nms_thresholds = model_info.get("FirstNMSThresholds")
                    first_nms_obj_thresholds = model_info.get("FirstNMSObjThresholds")
                    obj_thresholds = model_info.get("ObjThresholds")
                    val_med_lims = model_info.get("ValMedLims")
                    val_med_obj = model_info.get("ValMedObj")
                    prob_obj_cases = model_info.get("ProbObjCases")
                    print("Loaded model post-processing parameters.")
                    print("box_prior_class:", box_prior_class)
                    print("first_nms_thresholds:", first_nms_thresholds)
                    print("first_nms_obj_thresholds:", first_nms_obj_thresholds)
                    print("obj_thresholds:", obj_thresholds)
                    print("val_med_lims:", val_med_lims)
                    print("val_med_obj:", val_med_obj, "type:", type(val_med_obj))
                    print("prob_obj_cases:", prob_obj_cases)
                else:
                    logger.warning(f"[BATCH] no model info for {name}")

            except Exception:
                model_info = {}
            merged = {**model_info, **base_params, **p_job}

            try:
                patched_data, n_patches, info = normalize_and_patch_per_img(fits_file, p_client, merged)

            except Exception as e:
                logger.info(f"[BATCH] preprocess error {pid}: {e}")
                try:
                    update_job_status(pid, status="ERROR", comment=str(e),
                                      end_time=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                    shutil.move(job_dir, os.path.join(JOBS_ERROR, pid))
                except Exception as m:
                    logger.info(f"[BATCH] move-to-error failed {pid}: {m}")
                continue

            patch_data_list.append(patched_data)
            counts.append((pid, int(n_patches), job_dir))
            # display content of count
            logger.info(f"[BATCH] prepared {pid}: n_patches={n_patches}, job_dir={job_dir}")

            per_job_meta.append((pid, job_dir, info, merged))

        if not patch_data_list:
            logger.info("[BATCH] nothing to run (all prepro failed)")
            return

        _, _, _, merged0 = per_job_meta[0]

        nb_box   = as_int(merged0.get("YOLOBoxCount", 9)) # 
        nb_param = as_int(merged0.get("YOLOParamCount", 5))
        fwd_image_size = as_int(merged0.get("OriginalInputDim").split("x")[0], 512)
        c_size         = as_int(merged0.get("YOLOGridElemDim").split("x")[0], 16)
        yolo_nb_reg    = 16 # = int(fwd_image_size / c_size) forcage de la valeur

        per_patch_dim = nb_box * (8 + nb_param) * (yolo_nb_reg * yolo_nb_reg)

        # print("Batch forward parameters:")
        # print(f" nb_box: {nb_box}")
        # print(f" nb_param: {nb_param}")
        # print(f" fwd_image_size: {fwd_image_size}")
        # print(f" c_size: {c_size}")
        # print(f" yolo_nb_reg: {yolo_nb_reg}")
        # print(f" per_patch_dim: {per_patch_dim}")

        X_all = np.vstack(patch_data_list).astype(np.float32)
        total_patches = int(X_all.shape[0])
        logger.info(f"[BATCH] forward: total_patches={total_patches},\
                    feature_dim={X_all.shape[1]}")

        targets = np.zeros((total_patches, 1), dtype=np.float32)

        tmp_dir = make_batch_tmp_dir()
        with safe_chdir(tmp_dir):
            print("Before create_dataset")
            cianna.create_dataset("TEST", total_patches, X_all[:, :], targets[:, :])
            print("Before forward")
            cianna.forward(repeat=1, no_error=1, saving=2,
                           drop_mode="AVG_MODEL", silent=0)
            print("Forward done.")
            pred_file_path = os.path.join(tmp_dir, "fwd_res", "net0_0000.dat")

            if not os.path.exists(pred_file_path):
                cianna.delete_dataset("TEST")
                raise FileNotFoundError("forward output missing")

            predictions = np.fromfile(pred_file_path, dtype=np.float32)
            cianna.delete_dataset("TEST")

        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"[BATCH] tmp cleanup: {e}")

        total_vals = predictions.size
        print("Total predictions size:", predictions.shape)
        total_patches = sum(n for _, n, _ in counts)
        print("Total predictions values:", total_vals)
        print("Total patches processed:", total_patches)

         # Petit check rapide 
        if total_patches <= 0:
            raise ValueError(f"[BATCH] total_patches <= 0 (got {total_patches})")

        if total_vals % total_patches != 0:
            raise ValueError(
            f"[BATCH] inconsistent forward output size: "
            f"{total_vals} values for {total_patches} patches "
            f"(ratio={total_vals/total_patches:.4f}, not integer")

        per_patch_dim = total_vals // total_patches
        logger.info(
            f"[BATCH] forward output: total_vals={total_vals}, per_patch_dim={per_patch_dim}"
        )

        offset = 0
        logger.info(f"[BATCH] dispatching results")
        for (pid, n_patch, job_dir), (_, _, info, merged) in zip(counts, per_job_meta):
            try:
                n_vals = n_patch * per_patch_dim
                logger.info(
                    f"[BATCH] post-process {pid} with {n_patch} patches "
                    f"(slice length = {n_vals} = {n_patch} * {per_patch_dim})"
                )

                p = predictions[offset: offset + n_vals]
                if p.size != n_vals:
                    raise ValueError(
                        f"[BATCH] slice size mismatch for {pid}: "
                        f"got {p.size}, expected {n_vals}"
                    )
                offset += n_vals
                fwd_dir = os.path.join(job_dir, "fwd_res")
                os.makedirs(fwd_dir, exist_ok=True)
                dat_path = os.path.join(fwd_dir, f"net0_rts_{pid}.dat")
                p.astype(np.float32).tofile(dat_path)

                # Post-process + VOTable conversion
                vot_path = os.path.join(fwd_dir, f"net0_rts_{pid}.vot")
                try:
                    pred2votable(
                        dat_path=dat_path, img_info=info,
                        prob_obj_cases=prob_obj_cases, val_med_lims=val_med_lims,
                        val_med_obj=val_med_obj, box_prior_class=box_prior_class,
                        first_nms_thresholds=first_nms_thresholds,
                        first_nms_obj_thresholds=first_nms_obj_thresholds,
                        nb_box=nb_box, nb_param=nb_param,
                        yolo_nb_reg=yolo_nb_reg, votable_path=vot_path,
                    )
                except Exception as e:
                    logger.error(f"[BATCH] VOtable conversion error {pid}: {e}")
                    raise

                preview_votable(vot_path, n=5)

                update_job_status(pid, status="COMPLETED", comment="ok",
                                  end_time=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                shutil.move(job_dir, os.path.join(JOBS_COMPLETED, pid))
                finalize_job_id(model_path, pid)
                logger.info(f"[BATCH] done {pid}")
            except Exception as e:
                logger.error(f"[BATCH] finalize error {pid}: {e}")
                try:
                    update_job_status(pid, status="ERROR", comment=str(e),
                                      end_time=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                    shutil.move(job_dir, os.path.join(JOBS_ERROR, pid))
                    finalize_job_id(model_path, pid)
                except Exception as m:
                    logger.error(f"[BATCH] move-to-error failed {pid}: {m}")

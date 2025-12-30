import sys, os
import xml.etree.ElementTree as ET
import csv
import requests
from pathlib import Path
import numpy as np

import logging
logger = logging.getLogger(__name__)


# Chemin en dur... Ca me dégoute, mais c'est pour l'instant le plus simple
#XML_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/CIANNA_models.xml')

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
XML_MODEL_PATH = MODELS_DIR / "CIANNA_models.xml"  # Path, pas str


def process_xml(input_file, output_csv):
    try:
        # Check if the input file is empty
        with open(input_file, 'r') as f:
            if not f.read().strip():
                raise ValueError("The input file is empty or invalid.")
        
        # Parse the XML file
        tree = ET.parse(input_file)
        root = tree.getroot()

        # Extract fields with fallback for missing/empty tags
        user_id = root.find('USER_ID').text if root.find('USER_ID') is not None else 'N/A'
        timestamp = root.find('Timestamp').text if root.find('Timestamp') is not None else 'N/A'

        coordinates = root.find('Coordinates')
        ra = coordinates.find('RA').text if coordinates is not None and coordinates.find('RA') is not None else 'N/A'
        dec = coordinates.find('DEC').text if coordinates is not None and coordinates.find('DEC') is not None else 'N/A'
        h = coordinates.find('H').text if coordinates is not None and coordinates.find('H') is not None else 'N/A'
        w = coordinates.find('W').text if coordinates is not None and coordinates.find('W') is not None else 'N/A'

        image = root.find('Image').text if root.find('Image') is not None else 'N/A'
        quantization = root.find('Quantization').text if root.find('Quantization') is not None else 'N/A'

        # Write to CSV
        with open(output_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['USER_ID', 'Timestamp', 'RA', 'DEC', 'H', 'W', 'Image', 'Quantization', 'Status'])
            writer.writerow([user_id, timestamp, ra, dec, h, w, image, quantization, 'COMPLETED'])

    except ET.ParseError as e:
        logger.error(f"Error while processing the XML file: {e}")
        with open(output_csv, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Status'])
            writer.writerow(['ERROR'])

    except ValueError as e:
        logger.error(f"Error: {e}")
        with open(output_csv, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Status'])
            writer.writerow(['ERROR'])

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        with open(output_csv, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Status'])
            writer.writerow(['ERROR'])


def save_error_to_csv(output_csv, status):
    """
    Temporary function to handle error. 
    The future implementation should include more details about the error during
    the YOLO-CIANNA processing.
    """
    with open(output_csv, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", status])


def parse_job_parameters_xml(xml_path):
    """
    Parse the XML file and extract job parameters.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    user_id = root.find('USER_ID').text
    timestamp = root.find('Timestamp').text

    image = root.find('Image')
    coordinates = {
        'RA': float(image.find('RA').text),
        'DEC': float(image.find('DEC').text),
        'H': int(image.find('H').text),
        'W': int(image.find('W').text)
    }

    image_path = root.find('Image/path').text
    model = root.find('yolo_model/name').text
   
    filename = root.findtext('yolo_model/filename')
    quantization = root.findtext('preprocessing/quantization',  default="FP16C_FP32A")
    min_pix = float(root.findtext('preprocessing/min_pix', default=1e-7)) 
    max_pix = float(root.findtext('preprocessing/max_pix', default=1e-4))
    norm_fct = root.findtext('preprocessing/normalisation', default=["tanh"])

    # Make sure that norm_fct is a list
    if isinstance(norm_fct, str):
        norm_fct = [norm_fct]
    elif not isinstance(norm_fct, list):
        raise ValueError(f"Invalid type for norm_fct: {type(norm_fct)}. Expected str or list.")
    if not norm_fct:
        raise ValueError("norm_fct cannot be empty. Please provide at least one normalization function.")

   
    param_dict_client = {
        'user_id': user_id,
        'timestamp': timestamp,
        'coordinates': coordinates,
        'image_path': image_path,
        'model': model,
        'filename': filename,
        'quantization': quantization,
        'min_pix': min_pix,
        'max_pix': max_pix,
        'norm_fct': norm_fct,
        'fits_filename': os.path.basename(image_path) # used for local path inside EXECUTING
    }
    param_dict_model = get_model_info(XML_MODEL_PATH, model)
    if param_dict_model is None:
        raise ValueError(f"Model {model} not found in {XML_MODEL_PATH}")
    return param_dict_client, param_dict_model


def parse_float_list(value: str, default):
    if value is None:
        return np.array(default, dtype="float32")
    # "0.1, 0.2,0.3" → [0.1, 0.2, 0.3]
    parts = [v for v in value.replace(" ", "").split(",") if v]
    return np.array([float(v) for v in parts], dtype="float32")


def parse_int_list(value: str, default):
    if value is None:
        return np.array(default, dtype="int32")
    parts = [v for v in value.replace(" ", "").split(",") if v]
    return np.array([int(v) for v in parts], dtype="int32")


def parse_job_parameters(xml_path):
    """
    Parse a (possibly namespaced) UWS job XML into two dictionaries.

    Returns
    -------
    params_client : dict
        High-level job metadata (e.g., JobId, Phase, CreationTime, ExecutionDuration)
        plus a copy of FITS_Path if provided in the parameters.
        Keys follow your current capitalization (tag.capitalize()) for backward-compat.
    params_model : dict
        Model-related parameters extracted from <parameters>, including ModelName/ModelId,
        Quantization, ROI, RA/DEC, FITS_Path, etc. If the same parameter id appears
        multiple times, the value is a list in document order.
    """
    ns = {
        "uws": "http://www.ivoa.net/xml/UWS/v1.0",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _find_one(parent, tag):
        # Try with namespace, then without
        el = parent.find(f"uws:{tag}", ns)
        return el if el is not None else parent.find(tag)

    def _findall(parent, path):
        # Try namespaced path first, then non-namespaced fallback
        els = parent.findall(path, ns)
        if els:
            return els
        # crude fallback for simple "parameters/parameter" structure
        if path == "uws:parameters":
            return parent.findall("parameters")
        if path == "uws:parameters/uws:parameter":
            out = []
            for p in parent.findall("parameters"):
                out.extend(p.findall("parameter"))
            return out
        return []

    # ---------------- Client metadata ----------------
    params_client = {}
    for tag in ("jobId", "ownerId", "runId", "phase", "creationTime", "executionDuration"):
        el = _find_one(root, tag)
        if el is not None and el.text:
            # Preserve your existing capitalization style
            params_client[tag.capitalize()] = el.text.strip()

    # ---------------- Model parameters ----------------
    params_model = {}

    parameters_node = _find_one(root, "parameters")
    if parameters_node is not None:
        for param in _findall(root, "uws:parameters/uws:parameter"):
            key = param.attrib.get("id") or param.attrib.get("name")
            if not key:
                continue

            # Value can be text or byReference (xlink:href)
            val = (param.text or "").strip()
            by_ref = param.attrib.get("byReference")
            if (by_ref and by_ref.lower() == "true") or not val:
                href = param.attrib.get(f"{{{ns['xlink']}}}href")
                if href:
                    val = href

            # Merge duplicates into a list
            if key in params_model:
                if isinstance(params_model[key], list):
                    params_model[key].append(val)
                else:
                    params_model[key] = [params_model[key], val]
            else:
                params_model[key] = val

    # Convenience: mirror FITS_Path in params_client if present
    # (lets downstream code read it from either place)
    fits_key = None
    for k in ("FITS_Path", "FITSPath", "fits_path"):
        if k in params_model and isinstance(params_model[k], (str,)):
            fits_key = k
            break
    if fits_key:
        params_client["FITS_Path"] = params_model[fits_key]

    return params_client, params_model


def download_xml(url):
    """
    Download the XML content from the specified URL.

    Args:
        url (str): The URL from which to download the XML content.

    Returns:
        str or None: The XML content as a string if successful, otherwise None.
    """
    try:
        #logger.info("URL: {}".format(url))
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error("Error during download: %s", e)
        return None

def update_cianna_models(url: str, local_file: str | os.PathLike[str]) -> bool | None:
    """
    Download the remote XML file containing CIANNA models and update the local file,
    always retrieving the latest version without performing any version check.

    Args:
        url (str): The URL of the remote XML file.
        local_file (str): The path to the local file where the XML should be saved.

    Returns:
        bool or None: Returns True if the local file was updated successfully,
                      or None if the update failed.
    """

    remote_xml = download_xml(url)
    if remote_xml is None:
        logger.info("Failed to download remote file.")
        return None

    local_file = Path(local_file)
    target_name = local_file.name if local_file.name else "CIANNA_models.xml"
    out_path = MODELS_DIR / target_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(remote_xml, encoding="utf-8")
    logger.info("Model registry updated: %s", out_path)
    return True


def get_model_info(xml_path: str | os.PathLike[str], model_id: str) -> dict | None:
    """
    Retrieve model information either from a registry (with <Model>)
    or from a single UWS-like job file containing <uws:parameters>.
    Returns a dict or None if not found.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {"uws": "http://www.ivoa.net/xml/UWS/v1.0"}
    model_id_clean = Path(model_id).stem  # strip extension if any

    # Registry-style
    for model in root.findall("Model"):
        mid = (model.attrib.get("id") or "").strip()
        if Path(mid).stem == model_id_clean:
            info = {}
            for child in model:
                tag = (child.tag or "").strip()
                val = (child.text or "").strip() if child.text else None
                info[tag] = val
            return info

    # UWS-style fallback
    parameters_node = root.find("uws:parameters", ns) or root.find("parameters")
    if parameters_node is not None:
        info = {}
        params = parameters_node.findall("uws:parameter", ns) or parameters_node.findall("parameter")
        for param in params:
            key = param.attrib.get("id")
            val = (param.text or "").strip() if param.text else None
            if key:
                info[key] = val
        logger.info("[DEBUG] Parsed %d parameters from UWS-style model XML.", len(info))
        return info

    return None

#### NOUVELLE FONCTION APRES REFORMATAGE DE LA LOGIQUE DE BATCHING

def extract_model_and_quant(xml_path: str) -> tuple[str, str]:
    """
    Extract (model_id, quantization) from a client XML.
    Tries legacy tags first, then UWS <parameters>. Returns safe defaults if missing.
    """
    model = "default"
    quant = "FP16C_FP32A"  # garde ton défaut actuel ici

    try:
        root = ET.parse(xml_path).getroot()

        # Legacy-style
        m = root.findtext("yolo_model/filename")
        q = root.findtext("preprocessing/quantization")
        if m:
            model = m.strip()
        if q:
            quant = q.strip()

        # UWS <parameters>
        ns = {"uws": "http://www.ivoa.net/xml/UWS/v1.0", "xlink": "http://www.w3.org/1999/xlink"}
        params = root.find("uws:parameters", ns) or root.find("parameters")
        if params is not None:
            for p in params.findall("uws:parameter", ns) or params.findall("parameter"):
                pid = (p.attrib.get("id") or p.attrib.get("name") or "").lower()
                txt = (p.text or "").strip()
                if not txt:
                    # byReference
                    href = p.attrib.get("{http://www.w3.org/1999/xlink}href")
                    if href:
                        txt = href
                if pid in {"model", "modelname", "model_id", "filename"} and txt:
                    model = txt
                if pid in {"quant", "quantization"} and txt:
                    quant = txt

    except Exception as e:
        logger.warning("extract_model_and_quant: fallback defaults for %s (%s)", xml_path, e)

    return model or "default", quant or "FP16C_FP32A"



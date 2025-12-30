import requests
import xml.etree.ElementTree as ET
import os
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


# Namespace for UWS XML parsing
UWS_NS = {"uws": "http://www.ivoa.net/xml/UWS/v1.0"}

def download_xml(url):
    """
    Download the XML content from the specified URL.

    Args:
        url (str): The URL from which to download the XML content.

    Returns:
        str or None: The XML content as a string if successful, otherwise None.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error("[Download XML]❌ Error during download:", e)
        return None

def update_cianna_models(url: str, local_file: str, base_dir: str | None = None):
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
    logger.info("Downloading remote file...")

    remote_xml = download_xml(url)
    if remote_xml is None:
        logger.error("❌ Failed to download remote file.")
        return None


    if base_dir is None:
        base_dir_path = Path(__file__).resolve().parents[2]
    else:
        base_dir_path = Path(base_dir).resolve()

    p = Path(local_file)

    if not p.is_absolute():
        p = (base_dir_path / p).resolve()

    # Trick pour éviter le doublon '.../client/client/...'
    parts = []
    previous = None
    for part in p.parts:
        if previous == "client" and part == "client":
            # skip duplicate
            continue
        parts.append(part)
        previous = part
    p = Path(*parts)

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(remote_xml, encoding="utf-8")
    logger.info("✅ The local file has been updated to the latest version.")
    return True


def get_model_info_xml(xml_path, model_id):
    """
    Parses a CIANNA_models.xml file and retrieves the model information
    for a given model ID.

    Parameters
    ----------
    xml_path : str
        Path to the XML file that contains CIANNA model definitions.
    model_id : str
        The identifier of the model to retrieve (from the 'id' attribute in
        the <Model> tag).

    Returns
    -------
    dict or None
        A dictionary containing the model's parameters if found,
        otherwise None.
    """
    logger.info((f"Path for CIANNA models XML: {xml_path}"))
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for model in root.findall("Model"):
        if model.attrib.get("id") == model_id:
            info = {
                "Name": model.findtext("Name"),
                "ReleaseDate": model.findtext("ReleaseDate"),
                "BaseMemoryFootprint": model.findtext("BaseMemoryFootprint"),
                "OriginalInputDim": model.findtext("OriginalInputDim"),
                "PerImageMemoryFootprint": model.findtext("PerImageMemoryFootprint"),
                "MinInputDim": model.findtext("MinInputDim"),
                "MaxInputDim": model.findtext("MaxInputDim"),
                "YOLOGridElemDim": model.findtext("YOLOGridElemDim"),
                "YOLOBoxCount": int(model.findtext("YOLOBoxCount", default="8")),
                "YOLOParamCount": int(model.findtext("YOLOParamCount", default="5")),
                "YOLOGridCount": int(model.findtext("YOLOGridCount", default="16")),
                "DataNormalization": model.findtext("DataNormalization"),
                "DataQuantization": model.findtext("DataQuantization"),
                "TrainingQuantization": model.findtext("TrainingQuantization"),
                "InferenceQuantization": model.findtext("InferenceQuantization"),
                "InferenceMode": model.findtext("InferenceMode"),
                "InferencePatchShift": model.findtext("InferencePatchShift"),
                "ReceptiveField": model.findtext("ReceptiveField"),
                "CheckpointPath": model.findtext("CheckpointPath"),
                "Comments": model.findtext("Comments"),
            }
            return info
    return None 



def get_model_info(xml_path, model_id=None):
    """
    Parse a CIANNA_models.xml or UWS job XML file and return model info
    as a dictionary.
    """

    UWS_NS = {"uws": "http://www.ivoa.net/xml/UWS/v1.0"}

    logger.info((f"Path for CIANNA models XML: {xml_path}"))
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    tag = root.tag.lower()

    def _clean_text(t):
        return t.strip().strip('"').strip("'") if t else None

    # --- Case 1: UWS job format ---
    if "uws" in tag or "job" in tag:
        info = {}
        for p in root.findall(".//uws:parameter", namespaces=UWS_NS):
            pid = p.attrib.get("id")
            val = _clean_text(p.text)
            if pid:
                info[pid] = val
        # Add core job metadata
        info["jobId"] = root.findtext("uws:jobId", namespaces=UWS_NS)
        info["phase"] = root.findtext("uws:phase", namespaces=UWS_NS)
        info["creationTime"] = root.findtext("uws:creationTime", namespaces=UWS_NS)
        return info

    # --- Case 2: Legacy format ---
    for model in root.findall("Model"):
        if model.attrib.get("id") == model_id:
            info = {}
            for child in list(model):
                if child.text:
                    info[child.tag] = _clean_text(child.text)
            return info
    return None


#
# Functions compliant with UWS standard
#####################
def update_uws_job(url, local_file, verbose=False):
    """Download a remote UWS job XML and update the local copy."""
    if verbose:
        logger.info("Downloading remote UWS job file...")
        logger.info(f"URL: {url}")
        logger.info(f"Local file: {local_file}")

    remote_xml = download_xml(url)
    if remote_xml is None:
        if verbose:
            logger.error("Failed to download remote file.")
        return None

    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    with open(local_file, "w", encoding="utf-8") as f:
        f.write(remote_xml)
    if verbose:
        logger.info("Local UWS job file has been updated.")
    return True

def get_uws_parameters(xml_path):
    """
    Parses a UWS job XML file and retrieves all parameters
    defined in <uws:parameter id="..."> elements.

    Parameters
    ----------
    xml_path : str
        Path to the UWS XML file (e.g., job description).

    Returns
    -------
    dict
        Dictionary of all parameters: {id: value}.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    params = {}
    for param in root.findall(".//uws:parameter", namespaces=UWS_NS):
        param_id = param.attrib.get("id")
        param_value = param.text.strip() if param.text else None
        if param_id:
            params[param_id] = param_value

    return params

def get_uws_job_metadata(xml_path):
    """
    Extracts high-level job information such as jobId, phase, creationTime, etc.

    Parameters
    ----------
    xml_path : str

    Returns
    -------
    dict
        Dictionary with main job metadata.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    job_info = {
        "jobId": root.findtext("uws:jobId", default="", namespaces=UWS_NS),
        "runId": root.findtext("uws:runId", default="", namespaces=UWS_NS),
        "ownerId": root.findtext("uws:ownerId", default="", namespaces=UWS_NS),
        "phase": root.findtext("uws:phase", default="", namespaces=UWS_NS),
        "creationTime": root.findtext("uws:creationTime", default="", namespaces=UWS_NS),
        "executionDuration": root.findtext("uws:executionDuration", default="", namespaces=UWS_NS),
        "destruction": root.findtext("uws:destruction", default="", namespaces=UWS_NS),
    }

    return job_info

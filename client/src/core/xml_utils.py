import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from xml.sax.saxutils import escape
from xml.dom import minidom
import logging

logger = logging.getLogger(__name__)


def create_xml_param(params, fits_path=""):
    """
    Build a UWS-compliant XML job description for CIANNA OTF.
    args:
        params (dict): Dictionary containing parameters:
            - user_id (str): User identifier.
            - ra (float): Right Ascension in degrees.
            - dec (float): Declination in degrees.
            - w (int): Width of the bounding box in pixels.
            - h (int): Height of the bounding box in pixels.
            - image_path (str): Path to the FITS image file.
            - yolo_model (str): Name of the YOLO model to use.
            - quantization (str): Quantization type.
        fits_path (str): Path to the FITS file

    """
    UWS_NS = "http://www.ivoa.net/xml/UWS/v1.0"
    xsi_ns = "http://www.w3.org/2001/XMLSchema-instance"

    user_id = params.get("user_id", "unknown_user")
    name = params.get("name", "unknown_name")
    firstname = params.get("firstname", "unknown_firstname")
    ra_deg = float(params.get("ra", 0.0))
    dec_deg = float(params.get("dec", 0.0))
    width = int(params.get("w", 100))
    height = int(params.get("h", 100))
    fits_path = params.get("image_path", "")
    model_name = params.get("yolo_model", "default_model")
    quantization = params.get("quantization", "FP32C_FP32A")
    datenow = datetime.now(timezone.utc)
    datenow_str = datenow.isoformat(timespec='seconds')
    destruction_time = datenow + timedelta(hours=2)
    destruction_str = destruction_time.isoformat(timespec="seconds")

    ET.register_namespace("uws", UWS_NS)
    ET.register_namespace("xsi", xsi_ns)

    job = ET.Element(f"{{{UWS_NS}}}job")

    ET.SubElement(job, f"{{{UWS_NS}}}jobId").text = f"{user_id}"
    ET.SubElement(job, f"{{{UWS_NS}}}phase").text = "QUEUING"
    ET.SubElement(job, f"{{{UWS_NS}}}runId").text = model_name
    ET.SubElement(job, f"{{{UWS_NS}}}ownerId").text = str(firstname+"_"+name)
    ET.SubElement(job, f"{{{UWS_NS}}}creationTime").text = datenow_str
    ET.SubElement(job, f"{{{UWS_NS}}}executionDuration").text = "1800"
    ET.SubElement(job, f"{{{UWS_NS}}}destruction").text = destruction_str

    # Parameters block
    params_model = ET.SubElement(job, f"{{{UWS_NS}}}parameters")

    def add_param(pid, value):
        if value is None:
            return
        p = ET.SubElement(params_model, f"{{{UWS_NS}}}parameter", id=str(pid))
        p.text = str(value)

    # Core model parameters
    add_param("ModelName", model_name)
    add_param("ModelId", model_name)
    add_param("Quantization", quantization)
    add_param("RA", f"{ra_deg:.6f}")
    add_param("DEC", f"{dec_deg:.6f}")
    add_param("ROI_Width", width)
    add_param("ROI_Height", height)
    add_param("FITS_Path", fits_path)
    add_param("ClientVersion", "CIANNA_OTF_1.0")
    add_param("Timestamp", datenow.isoformat())

    # Results and error summary
    results = ET.SubElement(job, f"{{{UWS_NS}}}results")
    ET.SubElement(results, f"{{{UWS_NS}}}result", id="output")

    err = ET.SubElement(job, f"{{{UWS_NS}}}errorSummary",
                        type="transient", hasDetail="false")
    ET.SubElement(err, f"{{{UWS_NS}}}message").text = "No error yet (client-side request)."

    xml_bytes = ET.tostring(job, encoding="utf-8", xml_declaration=True)
    logger.debug("XML built for job %s", user_id)
    return xml_bytes.decode("utf-8")

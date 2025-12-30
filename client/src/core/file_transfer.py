import os, sys
import requests
import base64
import xml.etree.ElementTree as ET
from src.core.xml_utils import create_xml_param
from src.utils.tqdm_upfile import TqdmUploadFile
import logging

logger = logging.getLogger(__name__)


def send_xml_fits_to_server(server_url, xml_data):
    """
    Send XML and FITS file to the server under UWS-compliant /jobs/ endpoint.
    Compatible with both UWS and legacy XML formats.

    Args:
        server_url (str): Server base URL.
        xml_data (str): XML parameters as string.

    Returns:
        str: Job ID returned by the server, or None if failed.
    """

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        logger.error("[CLIENT] XML parsing error:", e)
        return None

    # --- Try to extract FITS path from different XML formats ------------------
    image_path = None
    try:
        # Case 1: new UWS XML format
        ns = {"uws": "http://www.ivoa.net/xml/UWS/v1.0"}
        node = root.find(".//uws:parameter[@id='FITS_Path']", namespaces=ns)
        if node is not None and node.text:
            image_path = node.text.strip()
        else:
            # Case 2: legacy format with <Image><path>...</path></Image>
            n2 = root.find(".//Image/path")
            if n2 is not None and n2.text:
                image_path = n2.text.strip()

        if not image_path:
            logger.error("[send_xml_fits_to_server] Could not find FITS path in XML (no <uws:parameter id='FITS_Path'> or <Image/path> tag).")
            return None
        if not os.path.exists(image_path):
            logger.error(f"[send_xml_fits_to_server] FITS file does not exist: {image_path}")
            return None

    except Exception as e:
        logger.error("[send_xml_fits_to_server] Error extracting FITS path from XML:", e)
        return None

    # --- Send the XML and FITS to the server ----------------------------------
    try:
        file_size = os.path.getsize(image_path)
        with open(image_path, "rb") as f:
            wrapped_file = TqdmUploadFile(f, total=file_size,
                                          desc=f"Uploading {os.path.basename(image_path)}")
            files = {
                "xml": ("parameters.xml", xml_data, "application/xml"),
                "fits": ("image.fits", wrapped_file, "application/octet-stream"),
            }
            response = requests.post(f"{server_url}/jobs/", files=files,
                                     allow_redirects=False)
    except Exception as e:
        logger.exception("[send_xml_fits_to_server] Error opening or sending file:", e)
        return None

    # --- Handle server response -----------------------------------------------
    if response.status_code == 303:
        location = response.headers.get('Location')
        if location:
            job_id = location.rstrip('/').split('/')[-1]
            logger.info(f"[send_xml_fits_to_server] 📡 Job submitted successfully. ID={job_id}")
            return job_id
    else:
        logger.info(f"[send_xml_fits_to_server] Server responded with {response.status_code}: {response.text}")
        return None
import os, sys
import random
import json
import requests
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import time
import argparse
from pathlib import Path
import logging

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "..", "..")))
from src.core.xml_utils import create_xml_param
from src.core.file_transfer import send_xml_fits_to_server
from src.services.server_comm import poll_for_completion, download_result
from src.utils.cianna_xml_updater import update_cianna_models, get_model_info
from src.utils.ssh_tunnel import create_ssh_tunnel
from src.utils.fits_utils import get_image_dim


# Pour la gestion des prints 
logging.basicConfig(
    level=logging.INFO,  # ou DEBUG pour plus de verbosité
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("emulate_client")


# Define directories
CLIENT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = CLIENT_ROOT / "configs"
MODELS_DIR = CLIENT_ROOT / "models"
RESULTS_DIR = CLIENT_ROOT / "results"
PREDICTIONS_DIR = CLIENT_ROOT / "predictions"
JOBS_SENT_DIR = CLIENT_ROOT / "JOBS_SENT"


TIME_POLLING = 5  # seconds between each poll for job completion

def load_config(config_path):
    """
    Load and parse a JSON configuration file.

    Parameters:
        config_path (str): The file path to the configuration JSON file.

    Returns:
        dict: A dictionary containing the configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from file {config_path}: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the config file {config_path}: {e}")
    
    return config


def emulate_client_request(server_url, image_path, config, params):
    """
    Simulates a client sending a request to a server with astronomical image data.

    This function generates randomized parameters for a simulated observation 
    (RA, DEC, bounding box dimensions), creates an XML-formatted request including 
    user ID and model configuration, and sends it to the specified server. It logs 
    the result of the request, including success or failure status.

    Parameters:
    ----------
    server_url : str
        The base URL of the server to which the request will be sent.
    image_path : str
        The path to the image file (typically a FITS file) to be sent.
    request_number : int
        A unique number used to differentiate this request and compute the user ID.
    config : dict
        A dictionary of configuration options, including:
            - "YOLO_MODEL" (str, optional): Name of the YOLO model file to use.
            - "QUANTIZATION" (str, optional): Type of quantization for the model.
    """

    yolo_model = params.get("yolo_model")
    user_id = params.get("user_id")

    ra = params.get("ra")
    dec = params.get("dec")
    h = params.get("h")
    w = params.get("w")

    # Check if the image is compatible with the YOLO model
    try:
        model_info = get_model_info(config.get("LOCAL_FILE_MODELS"), yolo_model)
    except Exception as e:
        logger.error(f"❌ Error reading model info: {e}")
        return

    if model_info is None:
        logger.error(f"❌ Error: Model {yolo_model} not found in the local XML file.")
        return

    else:
        # Get the image dimensions
        image_info = get_image_dim(image_path)
        if image_info is None:
            logger.error(f"❌ Error: Unable to read image dimensions from {image_path}.")
            return
        image_size = image_info.get('shape', (0, 0))
        if image_size[0] < h or image_size[1] < w:
            logger.error(f"❌ Error: Image dimensions {image_size} are smaller than the requested bounding box ({h}, {w}).")
            return

        xml_data = create_xml_param(params, fits_path=image_path)

        process_id = send_xml_fits_to_server(server_url, xml_data)
        logger.info(f"[EMULATE] Request sent. Process ID: {process_id}")
        if process_id is not None:

            try:
                # Poll for job completion of status
                phase = poll_for_completion(server_url, process_id, user_id, poll_interval=TIME_POLLING, timeout_s=600)
                if phase == "COMPLETED":
                    download_result(server_url, process_id, destination_folder=RESULTS_DIR)
                else:
                    logger.info(f"[CLIENT] Job {process_id} finished with phase={phase} (no download).")

            except requests.ConnectionError as e:
                logger.error(f"[EMULATE] Network error while polling/downloading: {e}")
            except requests.Timeout as e:
                logger.error(f"[EMULATE] Timeout error: {e}")
            except Exception as e:
                logger.exception(f"[EMULATE] Unexpected error: {e}")

def main():
    """
    Main function for the client.

    Steps:
      - Load the client configuration.
      - If a remote connection is specified, establish an SSH tunnel.
      - Update the local CIAnna models XML file by always retrieving the latest version.
      - Locate FITS images from the designated input folder.
      - Emulate a set number of client requests to the server.

    Note:
        - Vérifier sur le fichier FITS est compatible avec le modèle YOLO.
    """
    parser = argparse.ArgumentParser(description="Mockup client CIANNA_OTF.")
    parser.add_argument("--nb", type=int,
                        default=5, help="# of parallel requests (default: 5)")
    parser.add_argument("--interval", type=float,
                        default=1.0, help="Delay (in seconds) between request (default: 1.0)")
    args = parser.parse_args()

    nb_requests = args.nb
    interval = args.interval



    # Load configuration from JSON file
    config = load_config(os.path.join(CONFIGS_DIR,"param_cianna_rts_client.json"))
    
    # Path to the local Cianna models XML file
    local_models_file = config.get("LOCAL_FILE_MODELS")

    # Determine connection mode and set server URL accordingly.
    logger.info(40 * "-.")
    logger.info("\nemulate_client.py starting...")
    logger.info("Code to emulate multiple client requests to a CIANNA OTF server.\n")

    logger.info(40 * "-.")
    client_connexion = config.get("CLIENT_CONNEXION", "local").lower()
    tunnel = None
    if client_connexion == "remote":
        logger.info("🔗 Establishing remote connection via SSH tunnel...")
        tunnel = create_ssh_tunnel(
            ssh_server_ip = config.get("SSH_SERVER_IP"),
            ssh_username  = config.get("SSH_USERNAME"),
            ssh_password  = config.get("SSH_PASSWORD"),
            remote_port   = int(config.get("REMOTE_PORT", 5000)),
            local_port    = int(config.get("LOCAL_PORT", 5000))
        )
        server_url = f"http://127.0.0.1:{tunnel.local_bind_port}"
    else:
        logger.info("📡 Establishing local connection...")
        server_url = f"http://127.0.0.1:{config.get('LOCAL_PORT', 5000)}"
    
    logger.info(f"🔗 Connecting to a {client_connexion} server [{server_url}]...")

    # Update the local Cianna models XML file (always retrieves the latest version)
    models_url = f"{server_url}/model-files/CIANNA_models.xml"

    update_result = update_cianna_models(models_url, local_models_file)
    if update_result is None:
        logger.error("❌ Error updating CIANNA models.")
        if tunnel is not None:
            tunnel.stop()
        return

    # Get list if images for test
    image_folder = os.path.expanduser(config.get("IMAGE_FOLDER",
                                                 "/home/gsainton/01_CODES/DIR_images"))

    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".fits")]
    # images = ["/home/gsainton/01_CODES/DIR_images/RACS-DR1_0000+12A.fits"]
    
    logger.info(f"✅ Found {len(images)} FITS images.")

    if not images:
        logger.error("❌ No fits images in ", image_folder)
        if tunnel is not None:
            tunnel.stop()
        return

    max_workers = min(nb_requests, 8)
    logger.info(f"⚙️  Launching {nb_requests} parallel requests with up to {max_workers} workers…")
    name_list = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
    firstname_list = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda"]
    image_path = random.choice(images)



    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(nb_requests):
            name = name_list[i % len(name_list)]
            firstname = firstname_list[i % len(firstname_list)]
            datenow = datetime.now(timezone.utc).isoformat()
            params = {
                "name":name,
                "firstname":firstname,
                "datenow":datenow,
                "user_id":firstname[0].upper() + name.capitalize() + f"_{datenow}",
                "image_path":image_path,
                "ra": random.uniform(0, 360),     # Either the full image or a sub-image
                "dec": random.uniform(-90, 90),
                "h": random.randint(50, 200),
                "w":random.randint(50, 200),
                "yolo_model":"net0_s1800.dat",  # Supposed to be selected by the user 
                "quantization":"FP32C_FP32A",    # Supposed to be selected by the user
                "norm_fct":"tanh",                # Supposed to be selected by the user
                "request_number":i + 1
             }

            logger.info(f"\n⚙️ Scheduling request #{i+1} with image {image_path}")
            futures.append(
                executor.submit(
                    emulate_client_request,
                                server_url,
                                image_path,
                                config,
                                params
                )
            )
            if i < nb_requests - 1:
                    time.sleep(interval)


        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.exception(f"[EMULATE] ❌ A parallel request raised an error: {e}")
 
    # If an SSH tunnel was established, stop it after finishing
    if tunnel is not None:
        tunnel.stop()
        logger.info("❌ SSH tunnel closed.")

if __name__ == '__main__':
    main()
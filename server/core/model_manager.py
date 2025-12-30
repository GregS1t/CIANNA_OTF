# Define the model manager for handling CIANNA models

import time

import logging
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton-like manager for loading and managing a single CIANNA model at a time.

    Ensures that the model is only loaded once and reused across multiple jobs
    that require the same architecture and parameters.

    Attributes:
        current_model_path (str): Path of the currently loaded model.
        network (object): Loaded CIANNA model instance.
        last_used_time (float): Timestamp of last use (for possible future eviction strategy).
    """

    def __init__(self):
        self.current_model_path = None
        self.network = None
        self.last_used_time = 0

    def load_model(self, model_path):
        """
        Load the model from the specified path if it's not already loaded.

        Args:
            model_path (str): Path to the model file.

        Returns:
            object: The loaded CIANNA network object.
        """
        import CIANNA as cnn

        if self.current_model_path != model_path:
            try: 
                logger.info(f"[MODEL_MANAGER] 🔄 Loading new model: {model_path}")

                cnn.load(model_path)

                self.network = cnn
                self.current_model_path = model_path

            except Exception as e:
                logger.error(f"[MODEL_MANAGER] Failed to load model {model_path}: {e}")
                raise e
        else:
            logger.info(f"[MODEL_MANAGER] Reusing cached model: {model_path}")

        self.last_used_time = time.time()
        return self.network

    def get_loaded_model_path(self):
        """Returns the path of the currently loaded model."""
        return self.current_model_path

import tqdm
from flask import send_from_directory
import os


import logging
logger = logging.getLogger(__name__)

class TqdmUploadFile:
    def __init__(self, file_obj, total, desc="Uploading"): 
        self.file_obj = file_obj 
        self.total = total 
        self.tqdm_bar = tqdm(total=total, desc=desc, unit='B', unit_scale=True)


    def read(self, n):
        data = self.file_obj.read(n)
        self.tqdm_bar.update(len(data))
        return data

    # Delegate attribute access to the underlying file object
    def __getattr__(self, attr):
        return getattr(self.file_obj, attr)
    

def serve_models(filename):
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    return send_from_directory(models_dir, filename)
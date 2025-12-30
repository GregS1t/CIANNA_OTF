
from tqdm import tqdm

class TqdmUploadFile(object):
    def __init__(self, file, total=None, desc=None):
        self.file = file
        self.total = total or os.fstat(file.fileno()).st_size
        self.tqdm = tqdm(total=self.total, unit='B', unit_scale=True, desc=desc)
    def read(self, size=-1):
        data = self.file.read(size)
        self.tqdm.update(len(data))
        return data
    def __getattr__(self, name):
        return getattr(self.file, name)
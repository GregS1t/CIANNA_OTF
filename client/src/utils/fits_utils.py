import os
import pprint
from astropy.io import fits
from astropy.wcs import WCS

def get_image_dim(image_path):
    """
    Get the dimensions and type of a FITS image.
    
    Parameters
    ----------
    image_path : str
        Path to the FITS image file.
    
    Returns
    -------
    dict
        A dictionary containing the type, shape, and axes of the image.
    """

    # Open the image and get the header
    print("[CLIENT][get_image_dim] Opening image:", image_path)
    if not os.path.exists(image_path):
        print("[CLIENT][get_image_dim] Image file does not exist.")
        return None
    if not image_path.endswith('.fits'):
        print("[CLIENT][get_image_dim] Image file is not a FITS file.")
        return None

    with fits.open(image_path) as hdul:
        header = hdul[0].header
        data = hdul[0].data

        if data is None:
            return {'type': 'No data', 'shape': (), 'axes': [], 'ra_dec': None}

        shape = tuple(s for s in data.shape if s > 1) # Remove "1" dimensions
        ndim = len(shape)
        if ndim == 2:
            image_type = '2D image'
        elif ndim == 3:
            image_type = '3D image'
        else:
            image_type = f'Unsupported ({ndim}D)'
        wcs = WCS(header)
        try:
            while data.ndim > 2:
                 data = data[0]
            if wcs.naxis==4:
                wcs = wcs.dropaxis(2).dropaxis(2)

            ctype = [wcs.wcs.ctype[i] for i in range(wcs.naxis) if data.shape[::-1][i] > 1]
            center_pixel = [(s - 1) / 2.0 for s in shape[::-1]]
            ra_dec = wcs.all_pix2world([center_pixel], 0)[0][:2]
        except Exception:
            ctype = []
            ra_dec = None

        info = {
            'type': image_type,
            'shape': shape,
            'axes': ctype,
            'ra_dec': tuple(ra_dec) if ra_dec is not None else None
        }

        return info
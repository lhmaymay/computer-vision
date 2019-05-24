"""Download images from a dataset.

Downloads images, records unreadable image files.
"""
import urllib
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.externals.joblib import Parallel, delayed


def reshape(image,max_side_length):
    """Reshape an image such that the larger side is max_side_length.

    Used for scale normalization.

    Parameters
    ----------
    image : nd-array shape (width, height, 3) or (width,height) for grayscale
        Input image.
    max_side_length : maximum side length for the output image

    Returns
    -------
    image : nd-array
        Image with largest side max_side_length pixels.

    """
    # Use the OpenCV resize command
    mult = float(max_side_length) / max(image.shape[0], image.shape[1])
    # For some reason, the rows and cols are switched from the Mat definition
    new_size = (int(round(image.shape[1] * mult)), int(round(image.shape[0] * mult)))
    # If zooming the image (super-sampling)
    interpolation = cv2.INTER_LINEAR
    if( mult < 1.0 ):
        # If shrinking the image (sub-sampling)
        # This enables us to preserve, for instance, thin dark lines that show up in a white background
        interpolation = cv2.INTER_AREA
        
    return cv2.resize(image,new_size,interpolation=interpolation) 


def read_image(f, max_side_length=None):
    """Read an image the same way it is presented in QARES.

    Used for reading images everywhere.

    Parameters
    ----------
    f : string
        Path to image file
    max_side_length : integer
        max side length of rescaled output image (default is no rescaling)

    Returns
    -------
    image : nd-array, shape (width, height, 3)
        RGB image with maximum size max_side_length
    """
    if(f.endswith(".")):
        f = f + "jpg"
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Couldn't read image file %s " % f)

    # Grayscale images
    if(img.ndim == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        print("Original image is 2D %s" % f)

    # If the image is 16 bit convert to 8 bit
    if(img.dtype == 'uint16'):
        print("Convert from 16bit to 8bit %s" % f)
        img = (img >> 8).astype('uint8')

    # If there is an alpha channel, force the transparent pixels to be white.
    if(img.ndim == 3 and img.shape[2] == 4):
        alpha_mask = np.rollaxis(np.tile(img[:, :, 3], [3, 1, 1]), 0, 3).astype('float')
        img = (np.round(img[:, :, 0:3] * alpha_mask / 255.0 + (255.0 - alpha_mask))).astype('uint8')

    # OpenCV reads images as BGR, so we flip the colors
    if(not f.endswith(".gif") and img.ndim == 3):
        img = img[:, :, ::-1]

    if(max_side_length is None):
        return img
    else:
        return reshape(img, max_side_length)


def fetch_image_us(f, folder, max_side_length=None, min_max_side_length=None, check_corrupted=False):
    """Download a single image and store on disk.

    Used for collecting training images.

    Parameters
    ----------
    f : string
        Image file name.
    folder : string
        Folder to store the fetched data.
    max_side_length : int (default None means no resizing)
        Maximum image size to download
    min_max_side_length : int
        requires minimum side length of image to be at least min_max_side_length
    check_corrupted : bool
    """
    f = f.strip()
    if any(f.endswith(x) for x in ('.jpg', '.gif', '.png', '.', '.JPG', 'jpeg', 'tif', 'TIF', 'tiff', 'TIFF')):
        # figure out if we got a true url or a physical id
        prefix = ""
    else:
        f = f + ".jpg"
        prefix = "https://m.media-amazon.com/images/I/"
        #prefix = "http://ecx.images-amazon.com/images/I/"
    path = os.path.join(folder, f.split("/")[-1])
    if f.endswith("."):
        path = path + "jpg"
    url = prefix + f
    # Resize the image before downloading for more efficient download and storage
    if(max_side_length is not None):
        root, ext = os.path.splitext(url)
        url = root + '._SL' + str(max_side_length) + '_' + ext
    if not os.path.exists(path):
        # Ensure that file can be downloaded
        # Successful codes are 2xx
        success = urllib.request.urlopen(url).getcode() / 100 == 2
        if(not success):
            print("Couldn't open the url of the file %s" % (url))
            return False
        try:
            urllib.request.urlretrieve(url, path)
        except:
            print("Couldn't download the image %s" % (url))
            return False
        try:
            img = read_image(path, max_side_length)
            if min_max_side_length is not None and max(img.shape[:2]) < min_max_side_length:
                print("image %s has largest dimension smaller than %d" %(path, min_max_side_length))
                os.remove(path)
                return False
            elif check_corrupted and (np.all(img[-int(0.25 * img.shape[0]):, :, :] == 128) or np.all(img[-int(0.25 * img.shape[0]):, :, :] == 127)):
                print("image %s has corrupted gray stripe along bottom" % path)
                os.remove(path)
                return False
            else:
                return True
        except Exception as e:
            print("Can't read image %s: %s" % (path, e))
            if os.path.isfile(path):
                os.remove(path)
            return False
        # If the file already exists, then we successfully downloaded and
        # opened it in the past, so we just return True.
    return True


def download_data(folder, data, n_jobs=20, url_column_name = 'IMAGE_URL', subfolder_criteria = None, max_side_length = None, min_max_side_length = None, check_corrupted = False):
    """Download images from a dataset and store on disk. Used for collecting training images.

    Parameters
    ----------
    folder : string
        Folder to store the fetched data.
    data : pandas dataframe
        Dataframe with column IMAGE_URL which will be used to download images.
    n_jobs : int
    
    url_column_name : str
        name of column in data containing either image url or physical_id
    subfolder_criteria : str
        column of data to use in order to separate images into subfolders
    max_side_length : int, (default = None)
        for resizing images if not None
    min_max_side_length : int
        requires minimum side length of image to be at least min_max_side_length
    check_corrupted : bool
    """
    if not os.path.exists(os.path.join(folder, 'images')):
        os.makedirs(os.path.join(folder, 'images'))
        
    data = data.drop_duplicates(url_column_name)
    if subfolder_criteria is not None:
        assert subfolder_criteria in data.columns
        for s in data[subfolder_criteria].astype(str).unique():
            subfolder_str = ' '.join(s.split()).lower().replace(' ', '_')
            p = os.path.join(folder, 'images', subfolder_str)
            if not os.path.exists(p):
                os.mkdir(p)
        can_load = Parallel(n_jobs=n_jobs, verbose=10)(delayed(fetch_image_us)(f, os.path.join(folder, 'images', subfolder_str), max_side_length, min_max_side_length, check_corrupted) for (f,s) in zip(data[url_column_name], data[subfolder_criteria]))
    else:
        can_load = Parallel(n_jobs=n_jobs, verbose=10)(delayed(fetch_image_us)(f, os.path.join(folder, 'images'), max_side_length, min_max_side_length, check_corrupted) for f in data[url_column_name])
    
    image_urls_not_loadable = data.loc[~np.array(can_load), url_column_name]
    if(os.path.isfile(os.path.join(folder, "defective_image_urls.csv"))):
        old_defective_image_urls = pd.read_csv(os.path.join(folder, "defective_image_urls.csv"), header=None, names=['IMAGE_URL']).ix[:, 0]
        image_urls_not_loadable = pd.concat([image_urls_not_loadable, old_defective_image_urls])
    if not image_urls_not_loadable.empty:
        image_urls_not_loadable.drop_duplicates(inplace=True)
        image_urls_not_loadable.to_csv(os.path.join(folder, "defective_image_urls.csv"), index=False)
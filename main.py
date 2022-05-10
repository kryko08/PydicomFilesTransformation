import os

import numpy as np
import pydicom
from PIL import Image
from config import OUT_DIR, DCM_DIR_PATH, WINDOWS


# Applies slope and intercept from DICOM tags
def apply_slope_intercept(ds):
    array = ds.pixel_array.astype(float)
    try:
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
    except Exception:
        slope = 1
        intercept = 0
    if slope != 1 or intercept != 0:
        array = array * slope
        array = array + intercept
    return array


# Applies window transforamtions
def window(img, ww, wl):
    """
    Apply RTG window
    :param img: input image
    :param ww: window width
    :param wl: window level
    :return: new transformed array
    """
    upper, lower = wl+ww/2, wl-ww/2
    ar = np.clip(img, lower, upper)
    ar = ar - np.min(ar)
    ar = ar / np.max(ar)
    ar = (ar*255.0)
    ar = ar.astype(np.uint8)
    return ar


def write_3_channel_image(dicom_file, windows, target_directory):
    """
    This function takes dicom file, creates RGB image from pixel array and
    saves file to target directory
    :param dicom_file: Path to dicom file
    :param windows: RTG windows to apply
    :param target_directory: Directory to save new file into
    :return: None
    """

    ds = pydicom.dcmread(dicom_file)
    head, sop = os.path.split(dicom_file)
    jpg_file_name = sop + ".jpg"

    array = apply_slope_intercept(ds)

    # different width, level for each RGB channel
    brain_window = window(array, windows[0][0], windows[0][-1])
    subdural_window = window(array, windows[1][0], windows[1][-1])
    bone_window = window(array, windows[2][0], windows[2][-1])

    dummy = np.zeros((512, 512, 3), dtype=np.uint8)
    dummy[:, :, 0] = brain_window
    dummy[:, :, 1] = subdural_window
    dummy[:, :, 2] = bone_window

    image = Image.fromarray(dummy)

    dest_path = os.path.join(target_directory, jpg_file_name)
    image.save(dest_path)


if __name__ == "__main__":
    # List all images in DCM_DIR_PATH
    dcm_dir = os.listdir(DCM_DIR_PATH)
    for dcm_file in dcm_dir:
        sop = dcm_file
        file_path = os.path.join(DCM_DIR_PATH, dcm_file)
        write_3_channel_image(file_path, WINDOWS, OUT_DIR)



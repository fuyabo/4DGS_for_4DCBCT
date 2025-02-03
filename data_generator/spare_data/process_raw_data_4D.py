""" 
Process raw CT data.
1. Normalize to [0, 1] and a cube.
2. Save as *.npy format.
"""

import os
import os.path as osp
import numpy as np
import pydicom
import argparse
import glob
import scipy.ndimage as ndimage
from tqdm import tqdm, trange
import sys
import tifffile
import importlib
from scipy.io import loadmat

sys.path.append("./")


def main():
    output_path = 'C:/R2_Gaussian_4D/data_generator/spare_data/processed'
    os.makedirs(output_path, exist_ok=True)
    case = 'mc_p1_ns'
    p = f'C:/R2_Gaussian_4D/data_generator/spare_data/raw/{case}'
    for t in range(10):
        t_t = str(t+1).zfill(2)
        t_path = osp.join(p, f"gt_{t_t}.mat")
        case_output_path = osp.join(output_path, f"{case}_{t_t}.npy")
        vol = loadmat(t_path)
        vol = vol['image']
        vol_min = vol.min()
        vol_max = vol.max()
        vol = (vol - vol_min) / (vol_max - vol_min)
        # voxel_spacing = np.array([1, 1, 1])
        np.save(case_output_path, vol.astype(np.float32))


def process_dcm(dcm_fd, target_size):
    """Process *.dcm files."""
    dcm_path = sorted(glob.glob(osp.join(dcm_fd, "*.dcm")))
    slices = []
    for d_path in dcm_path:
        ds = pydicom.dcmread(d_path)
        slice = np.array(ds.pixel_array).astype(float) * float(ds.RescaleSlope) + float(
            ds.RescaleIntercept
        )
        slices.append(slice)
    vol = np.stack(slices, axis=-1)
    vol = vol[:, :, ::-1]  # Upside down
    vol = vol.clip(-1000, 2000)  # From air to bone
    vol_min = vol.min()
    vol_max = vol.max()
    vol = (vol - vol_min) / (vol_max - vol_min)
    slice_thickness = ds.SliceThickness
    pixel_spacing = [float(i) for i in list(ds.PixelSpacing)]
    voxel_spacing = np.array(pixel_spacing + [slice_thickness])
    vol_new = reshape_vol(vol, voxel_spacing, target_size, None)
    vol_new = vol_new.clip(0.0, 1.0)
    return vol_new


def process_raw(case_info, target_size):
    """Process *.raw file."""
    data = (
        np.fromfile(case_info["raw_path"], dtype=case_info["dtype"])
        .reshape(case_info["shape"][::-1])
        .astype(float)
    )
    data = data.transpose([2, 1, 0])
    data_min = data.min()
    data_max = data.max()
    data = (data - data_min) / (data_max - data_min)
    data = data.clip(0.0, 1.0)
    data = reshape_vol(
        data, case_info["spacing"], target_size, mode=case_info["reshape"]
    )
    data = data.clip(0.0, 1.0)
    data = data.transpose(case_info["transpose"])
    if case_info["z_invert"]:
        data = data[:, :, ::-1]
    return data


def process_tif(case_info, target_size):
    """Process *.tif file."""
    data = tifffile.imread(case_info["raw_path"])
    data_min = data.min()
    data_max = data.max()
    data = (data - data_min) / (data_max - data_min)
    data = reshape_vol(
        data, case_info["spacing"], target_size, mode=case_info["reshape"]
    )
    data = data.clip(0.0, 1.0)
    data = data.transpose(case_info["transpose"])
    if case_info["z_invert"]:
        data = data[:, :, ::-1]
    return data


def reshape_vol(image, spacing, target_size, mode=None):
    """Reshape a CT volume."""

    if mode is not None:
        image, _ = resample(image, spacing, [1, 1, 1])
        if mode == "crop":
            image = crop_to_cube(image)
        elif mode == "expand":
            image = expand_to_cube(image)
        else:
            raise ValueError("Unsupported reshape mode!")

    image_new = resize(image, target_size)
    return image_new


def expand_to_cube(array):
    # Step 1: Find the maximum dimension
    max_dim = max(array.shape)

    # Step 2: Calculate the padding for each dimension
    padding = [(max_dim - s) // 2 for s in array.shape]
    # For odd differences, add an extra padding at the end
    padding = [(pad, max_dim - s - pad) for pad, s in zip(padding, array.shape)]

    # Step 3: Pad the array to get the cubic shape
    cubic_array = np.pad(
        array, padding, mode="constant", constant_values=0
    )  # Using zero padding

    return cubic_array


def crop_to_cube(array):
    # Step 1: Find the minimum dimension
    min_dim = min(array.shape)

    # Step 2: Define the start and end indices for cropping
    start_indices = [(dim_size - min_dim) // 2 for dim_size in array.shape]
    end_indices = [start + min_dim for start in start_indices]

    # Step 3: Crop the array to get the cubic region
    cubic_region = array[
        start_indices[0] : end_indices[0],
        start_indices[1] : end_indices[1],
        start_indices[2] : end_indices[2],
    ]

    return cubic_region


def resample(image, spacing, new_spacing=[1, 1, 1]):
    """Resample to stantard spacing (keep physical scale stable, change pixel numbers)"""
    # .mhd image order : z, y, x
    if not isinstance(spacing, np.ndarray):
        spacing = np.array(spacing)
    if not isinstance(new_spacing, np.ndarray):
        new_spacing = np.array(new_spacing)

    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = ndimage.zoom(image, real_resize_factor, mode="nearest")
    return image, new_spacing


def resize(scan, target_size):
    """Resize the scan based on given voxel dimension."""
    scan_x, scan_y, scan_z = scan.shape
    zoom_x = target_size / scan_x
    zoom_y = target_size / scan_y
    zoom_z = target_size / scan_z

    if zoom_x != 1.0 or zoom_y != 1.0 or zoom_z != 1.0:
        scan = ndimage.zoom(
            scan,
            (zoom_x, zoom_y, zoom_z),
            mode="nearest",
        )
    return scan


if __name__ == "__main__":
    main()

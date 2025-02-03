import os
import numpy as np
import tigre.algorithms as algs
import open3d as o3d
import sys
import argparse
import os.path as osp
import json
import pickle
from tqdm import trange
import copy
import torch

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry, recon_volume
from r2_gaussian.arguments import ParamGroup, ModelParams, PipelineParams
from r2_gaussian.utils.plot_utils import show_one_volume, show_two_volume
from r2_gaussian.gaussian import GaussianModel, query, initialize_gaussian
from r2_gaussian.utils.image_utils import metric_vol
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.general_utils import t2a
import glob

np.random.seed(0)

import numpy as np
from scipy.spatial import cKDTree

def remove_close_points(points, distance_threshold):
    """
    Remove points that are too close to each other.

    Args:
        points (np.ndarray): Array of shape [n, 4], where the first 3 columns are x, y, z coordinates.
        distance_threshold (float): Minimum allowable distance between points.

    Returns:
        np.ndarray: Filtered array of points with close points removed.
    """
    # Extract the x, y, z coordinates
    xyz = points[:, :3]

    # Use cKDTree for efficient spatial searching
    tree = cKDTree(xyz)

    # Find all pairs of points within the distance threshold
    pairs = tree.query_pairs(distance_threshold)

    # Convert pairs to a set of indices to remove
    remove_indices = set()
    for i, j in pairs:
        # Add the second point to the removal set
        remove_indices.add(j)

    # Keep points not in the removal set
    mask = np.array([i not in remove_indices for i in range(points.shape[0])])

    return points[mask]

def init_pcd_grid(
        projs,
        angles,
        geo,
        scanner_cfg,
        recon_method,
        density_thresh,
        density_rescale,
        grid_spacing,
        save_path,
        vol = None,
        recon_file_path = None,
):
    "Initialize Gaussians with grid-based sampling."

    # Use traditional algorithms for initialization
    print(
        f"Initialize point clouds with the volume reconstructed from {recon_method}."
    )
    if vol is None:
        vol = recon_volume(projs, angles, copy.deepcopy(geo), recon_method)
        np.save(recon_file_path, vol)

    show_one_volume(vol)

    density_mask = vol > density_thresh
    offOrigin = np.array(scanner_cfg["offOrigin"])
    dVoxel = np.array(scanner_cfg["dVoxel"])
    sVoxel = np.array(scanner_cfg["sVoxel"])

    # Grid sampling: Generate every n-th voxel position
    grid_indices = np.array(np.meshgrid(
        np.arange(0, vol.shape[0], grid_spacing),
        np.arange(0, vol.shape[1], grid_spacing),
        np.arange(0, vol.shape[2], grid_spacing),
        indexing='ij'
    )).reshape(3, -1).T  # Shape: (num_samples, 3)

    # Filter grid indices using the density mask
    sampled_indices = grid_indices[density_mask[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]]
    print(f" sampled {sampled_indices.shape[0]} points")
    # Convert sampled indices to physical positions
    sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin

    # Get densities at sampled positions
    sampled_densities = vol[
        sampled_indices[:, 0],
        sampled_indices[:, 1],
        sampled_indices[:, 2],
    ]
    sampled_densities = sampled_densities * density_rescale
    out_grid = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    # sample some at the edge
    dx, dy, dz = np.gradient(vol)  # Compute gradients along each axis
    gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) * 1000  # Compute gradient magnitude

    edge_mask = gradient_magnitude > 1.5
    edge_mask = mask_cylinder(edge_mask, 50)
    edge_mask[:,:, :10] = 0
    edge_mask[:,:, 210:] = 0
    edge_mask = edge_mask & density_mask
    show_one_volume(edge_mask)
    valid_indices = np.argwhere(edge_mask)
    n_points = 20000
    assert (
            valid_indices.shape[0] >= n_points
    ), "Valid voxels less than target number of sampling. Check threshold"
    offOrigin = np.array(scanner_cfg["offOrigin"])
    dVoxel = np.array(scanner_cfg["dVoxel"])
    sVoxel = np.array(scanner_cfg["sVoxel"])

    sampled_indices = valid_indices[
        np.random.choice(len(valid_indices), n_points, replace=False)
    ]
    sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
    sampled_densities = vol[
        sampled_indices[:, 0],
        sampled_indices[:, 1],
        sampled_indices[:, 2],
    ]
    sampled_densities = sampled_densities * density_rescale

    out_edge = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    # Save the result
    out = np.concatenate([out_grid, out_edge])
    out = remove_close_points(out, 0.01)
    # out = out_edge
    np.save(save_path, out)
    print(f"Initialization saved in {save_path}.")


def mask_cylinder(image, margin):
    # Cylinder parameters
    shape = image.shape
    radius = image.shape[0]/2 - margin
    center_x, center_y = shape[0] / 2, shape[1] / 2
    length = shape[2]

    # Create meshgrids for x, y, and z coordinates
    x = np.arange(shape[0]) - center_x
    y = np.arange(shape[1]) - center_y
    z = np.arange(length)

    # Create 3D meshgrid for x and y
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Calculate distance from the cylinder's central axis in the xy-plane
    distance_from_center = np.sqrt(xx**2 + yy**2)

    # Create the cylindrical mask
    cylinder_mask = distance_from_center <= radius

    # Apply mask to a sample 3D array (e.g., mask out values outside the cylinder)
    masked_image = np.where(cylinder_mask, image, 0)  # Mask out values outside the cylinder
    return masked_image


class InitParams(ParamGroup):
    def __init__(self, parser):
        self.recon_method = "cgls"
        self.n_points = 250000
        self.density_thresh = 0.005
        self.density_rescale = 0.05
        super().__init__(parser, "Initialization Parameters")


def init_pcd(
    projs,
    angles,
    geo,
    scanner_cfg,
    recon_method,
    n_points,
    density_thresh,
    density_rescale,
    save_path,
):
    "Initialize Gaussians."
    assert recon_method in ["random", "fdk", "cgls"], "--recon_method not supported."
    if recon_method == "random":
        print(f"Initialize random point clouds.")
        sampled_positions = geo.offOrigin + geo.sVoxel * (
            np.random.rand([n_points, 3]) - 0.5
        )
        sampled_densities = np.random.rand(
            [
                n_points,
            ]
        )
    else:
        # Use traditional algorithms for initialization
        print(
            f"Initialize point clouds with the volume reconstructed from {recon_method}."
        )
        vol = recon_volume(projs, angles, copy.deepcopy(geo), recon_method)
        show_one_volume(vol)
        density_mask = vol > density_thresh
        valid_indices = np.argwhere(density_mask)
        offOrigin = np.array(scanner_cfg["offOrigin"])
        dVoxel = np.array(scanner_cfg["dVoxel"])
        sVoxel = np.array(scanner_cfg["sVoxel"])

        assert (
            valid_indices.shape[0] >= n_points
        ), "Valid voxels less than target number of sampling. Check threshold"

        sampled_indices = valid_indices[
            np.random.choice(len(valid_indices), n_points, replace=False)
        ]
        sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
        sampled_densities = vol[
            sampled_indices[:, 0],
            sampled_indices[:, 1],
            sampled_indices[:, 2],
        ]
        sampled_densities = sampled_densities * density_rescale

    out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    np.save(save_path, out)
    print(f"Initialization saved in {save_path}.")


def main(
    args, init_args: InitParams, model_args: ModelParams, pipe_args: PipelineParams
):
    # Read scene
    data_path = args.data
    model_args.source_path = data_path
    scene = Scene(model_args, False)  #! Here we scale the scene to [-1,1]^3 space.
    train_cameras = scene.getTrainCameras()
    projs_train = np.concatenate(
        [t2a(cam.original_image) for cam in train_cameras], axis=0
    )
    angles_train = np.stack([t2a(cam.angle) for cam in train_cameras], axis=0)
    scanner_cfg = scene.scanner_cfg
    geo = get_geometry(scanner_cfg)
    geo.filter = 'ram_lak'
    save_path = args.output
    if not save_path:
        save_path = osp.join(
            data_path, "init_" + osp.basename(data_path).split(".")[0] + ".npy"
        )
    assert not osp.exists(
        save_path
    ), f"Initialization file {save_path} exists! Delete it first."
    os.makedirs(osp.dirname(save_path), exist_ok=True)

    cgls_file_path = osp.join(
            data_path, "cgls.npy"
        )
    if os.path.exists(cgls_file_path):
        # Load the file using numpy
        cgls = np.load(cgls_file_path)
        print("cgls file loaded successfully.")
    else:
        print("cgls file does not exist... need recon ...")
        cgls = None

    init_pcd_grid(
        projs_train,
        angles_train,
        geo,
        scanner_cfg,
        'cgls',
        density_thresh = 0.005,
        density_rescale = 0.05, # empirical
        grid_spacing = 8,
        save_path = save_path,
        vol = cgls,
        recon_file_path = cgls_file_path,
    )

    # Evaluate using ground truth volume (for debug only)
    if args.evaluate:
        with torch.no_grad():
            model_args.ply_path = save_path
            scale_bound = None
            volume_to_world = max(scanner_cfg["sVoxel"])
            if model_args.scale_min and model_args.scale_max:
                scale_bound = (
                    np.array([model_args.scale_min, model_args.scale_max])
                    * volume_to_world
                )
            gaussians = GaussianModel(scale_bound)
            initialize_gaussian(gaussians, model_args, None)
            vol_pred = query(
                gaussians,
                scanner_cfg["offOrigin"],
                scanner_cfg["nVoxel"],
                scanner_cfg["sVoxel"],
                pipe_args,
            )["vol"]
            vol_gt = scene.vol_gt.cuda()
            psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
            print(f"3D PSNR for initial Gaussians: {psnr_3d}")
            # show_two_volume(vol_gt, vol_pred, title1="gt", title2="init")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate initialization parameters")
    init_parser = InitParams(parser)
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--data", type=str, help="Path to data.")
    parser.add_argument("--output", default=None, type=str, help="Path to output.")
    parser.add_argument("--evaluate", default=False, action="store_true", help="Add this flag to evaluate quality (given GT volume, for debug only)")
    # fmt: on

    args = parser.parse_args()

    folder_path = 'Z:\Yabo\SPARE/raw\MonteCarloDatasets'
    file_list = glob.glob(os.path.join('C:\R2_Gaussian_4D\data/spare_dataset', "*"), recursive=False)
    file_list = [f for f in file_list if os.path.basename(f).startswith('MC')]

    for f in file_list:
        if os.path.exists(os.path.join(f, 'init_{}.npy'.format(os.path.basename(f)))):
            print("init pc already exist!")
            continue
        else:
            args.data = f
            args.output = os.path.join(f, "init_{}.npy".format(os.path.basename(f)))
            print("processing {}".format(f))
            main(args, init_parser.extract(args), lp.extract(args), pp.extract(args))

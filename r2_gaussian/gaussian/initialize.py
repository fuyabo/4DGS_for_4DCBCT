import os
import sys
import os.path as osp
import numpy as np

sys.path.append("./")
from r2_gaussian.gaussian.gaussian_model import GaussianModel
from r2_gaussian.arguments import ModelParams
from r2_gaussian.utils.graphics_utils import fetchPly
from r2_gaussian.utils.system_utils import searchForMaxIteration
from scipy.spatial import KDTree


def eliminate_close_points(points, min_distance):
    """
    Eliminate points that are too close to each other.

    Parameters:
        points (numpy.ndarray): Array of shape [n, 4], where the first 3 columns are x, y, z coordinates.
        min_distance (float): Minimum allowable distance between points.

    Returns:
        numpy.ndarray: Filtered array with points spaced at least min_distance apart.
    """
    # Create a KDTree for efficient spatial queries
    kdtree = KDTree(points[:, :3])

    # List to keep the indices of retained points
    retained_indices = []
    visited = np.zeros(len(points), dtype=bool)

    for i, point in enumerate(points):
        if not visited[i]:  # Check if the point is already processed
            retained_indices.append(i)
            neighbors = kdtree.query_ball_point(point[:3], min_distance)
            visited[neighbors] = True  # Mark all neighbors as visited

    return points[retained_indices]

def initialize_gaussian(gaussians: GaussianModel, args: ModelParams, loaded_iter=None):
    if loaded_iter:
        if loaded_iter == -1:
            loaded_iter = searchForMaxIteration(
                osp.join(args.model_path, "point_cloud")
            )
        print("Loading trained model at iteration {}".format(loaded_iter))
        gaussians.load_ply(
            os.path.join(
                args.model_path,
                "point_cloud",
                "iteration_" + str(loaded_iter),
                "point_cloud.ply",
            )
        )
        # load deformation model
        gaussians.load_deformation_model(os.path.join(
                args.model_path,
                "point_cloud",
                "iteration_" + str(loaded_iter)
            ))
    else:
        if not args.ply_path:
            if osp.exists(osp.join(args.source_path, "meta_data.json")):
                ply_path = osp.join(
                    args.source_path, "init_" + osp.basename(args.source_path) + ".npy"
                )
            elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
                ply_path = osp.join(
                    osp.dirname(args.source_path),
                    "init_" + osp.basename(args.source_path).split(".")[0] + ".npy",
                )
            else:
                assert False, "Could not recognize scene type!"
        else:
            ply_path = args.ply_path

        assert osp.exists(
            ply_path
        ), "Cannot load point cloud for initialization. Please specify a valid ply_path or generate point cloud with initialize_pcd.py."

        print(f"Initialize Gaussians with {osp.basename(ply_path)}")
        ply_type = ply_path.split(".")[-1]
        if ply_type == "npy":
            point_cloud = np.load(ply_path)
            point_cloud = eliminate_close_points(point_cloud, 0.01)
            xyz = point_cloud[:, :3]
            density = point_cloud[:, 3:4]
        elif ply_type == ".ply":
            point_cloud = fetchPly(ply_path)
            xyz = np.asarray(point_cloud.points)
            density = np.asarray(point_cloud.colors[:, :1])

        gaussians.create_from_pcd(xyz, density, 1.0)

    return loaded_iter

import os
import os.path as osp
import tigre
from tigre.utilities.geometry import Geometry
from tigre.utilities import gpu
import numpy as np
import yaml
import plotly.graph_objects as go
import scipy.ndimage.interpolation
from tigre.utilities import CTnoise
import json
import matplotlib.pyplot as plt
import tigre.algorithms as algs
import argparse
import open3d as o3d
import cv2
import pickle
import copy

import sys

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry, recon_volume


def categorize_angles(angles, num_groups=10, per_phase_num_frames=5):
    group_dict = {str(i * 10).zfill(2): [] for i in range(num_groups)}

    # Loop through each angle, assigning to groups in blocks of per_phase_num_frames
    for i in range(0, len(angles), per_phase_num_frames):
        # Determine the group index based on the position in the list
        group_index = (i // per_phase_num_frames) % num_groups  # Wrap around every 10 groups
        t = str(group_index * 10).zfill(2)
        # Assign the current block of angles to the correct group
        group_dict[t].extend(angles[i:i + per_phase_num_frames])

    return group_dict


def split_train_test(angle, projs):
    n = angle.shape[0]
    indices = np.random.permutation(n)

    # Calculate the split index
    split_index = int(n * 0.9)

    # Split the indices into 90% and 10%
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    # Split a and b into 90% (a, b) and 10% (c, d)
    angle_train = angle[train_indices]
    projs_train = projs[train_indices]
    angle_test = angle[test_indices]
    projs_test = projs[test_indices]
    return angle_train, projs_train, angle_test, projs_test


def main(args):
    """Assume CT is in a unit cube. We synthesize X-ray projections."""
    vol_path = args.vol
    scanner_cfg_path = args.scanner
    n_train = args.n_train
    n_test = args.n_test
    vol_name = osp.basename(vol_path)[:-4]
    output_path = args.output

    # Load configuration
    with open(scanner_cfg_path, "r") as handle:
        scanner_cfg = yaml.safe_load(handle)

    case_name = f"{vol_name}_{scanner_cfg['mode']}"
    print(f"Generate data for case {case_name}")

    case_save_path = osp.join(output_path, case_name)
    os.makedirs(case_save_path, exist_ok=True)

    geo = get_geometry(scanner_cfg)
    file_path_dict = {}
    gantry_rotate_speed = 360 / 60. # 6 degree per second
    resp_cycle_second = 4 # 4 second
    resp_cycle_degree = gantry_rotate_speed * resp_cycle_second #24 degree
    per_phase_degree = resp_cycle_degree / 10 # 2.4 degree
    cbct_fps = 15
    per_phase_second = per_phase_degree / gantry_rotate_speed
    per_phase_num_frames = round(per_phase_second * cbct_fps)
    frame_interval_degree = 6./15.
    num_of_phases = 10


    file_path_dict['proj_train'] = []
    file_path_dict['proj_test'] = []
    projs_train_angles = np.arange(0, 360 + frame_interval_degree, frame_interval_degree)
    grouped_angles = categorize_angles(projs_train_angles, num_of_phases, per_phase_num_frames)

    # Load volume
    vol_gt_list = []
    vol_4d_dict = {}
    for t in range(10):
        t_t = str(t * 10).zfill(2)
        vol = np.load(os.path.join(vol_path, f'ct_{t_t}.npy')).astype(np.float32)
        vol_gt_list.append(f"vol_gt_{t_t}.npy")
        vol_4d_dict[t_t] = vol
        np.save(osp.join(case_save_path, f"vol_gt_{t_t}.npy"), vol)

    c=0
    for group, angles in grouped_angles.items():
        print(f"{group}: {angles}")
        angles = np.array(angles)
        angles_radian = np.pi * angles / 180.
        t_vol = vol_4d_dict[group]
        t = int(group)/100
        projs_train = tigre.Ax(
            np.transpose(t_vol, (2, 1, 0)).copy(), geo, angles_radian
        )[:, ::-1, :]
        if scanner_cfg["noise"]:
            projs_train = CTnoise.add(
                projs_train,
                Poisson=scanner_cfg["possion_noise"],
                Gaussian=np.array(scanner_cfg["gaussian_noise"]),
            )  #
            projs_train[projs_train < 0.0] = 0.0

        # random split 10% as testing images
        projs_train_angles, projs_train, projs_test_angles, projs_test = split_train_test(angles_radian, projs_train)

        # recon this phase with the trainig projections using FDK for comparison purpose
        geo = get_geometry(scanner_cfg)
        ct_fdk = algs.fdk(projs_train, geo, projs_train_angles)
        ct_fdk = ct_fdk.transpose((2, 1, 0))
        ct_fdk[ct_fdk < 0] = 0
        np.save(osp.join(case_save_path, f"vol_fdk_{group}.npy"), ct_fdk)

        #
        if False:
            for split, projs, angles in zip(
                ["proj_train", "proj_test"],
                [projs_train, projs_test],
                [projs_train_angles, projs_test_angles],
            ):
                os.makedirs(osp.join(case_save_path, split), exist_ok=True)

                for i_proj in range(projs.shape[0]):
                    proj = projs[i_proj]
                    frame_save_name = osp.join(split, f"{split}_{c}.npy")
                    c = c + 1
                    np.save(osp.join(case_save_path, frame_save_name), proj)
                    file_path_dict[split].append(
                        {
                            "file_path": frame_save_name,
                            "angle": angles[i_proj],
                            "time": t
                        }
                    )
    if False:
        meta = {
            "scanner": scanner_cfg,
            "vol": vol_gt_list,
            "bbox": [[-1, -1, -1], [1, 1, 1]],
            "proj_train": file_path_dict["proj_train"],
            "proj_test": file_path_dict["proj_test"],
        }
        with open(osp.join(case_save_path, "meta_data.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
    print(f"Generate data for case {case_name} complete!")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Data generator parameters")
    
    parser.add_argument("--vol", default="data_generator/volume_gt/0_chest.npy", type=str, help="Path to volume.")
    parser.add_argument("--scanner", default="data_generator/scanner/cone_beam.yml", type=str, help="Path to scanner configuration.")
    parser.add_argument("--output", default="data/cone_ntrain_50_angle_360", type=str, help="Path to output.")
    parser.add_argument("--n_train", default=50, type=int, help="Number of projections for training.")
    parser.add_argument("--n_test", default=100, type=int, help="Number of projections for evaluation.")
    # fmt: on

    args = parser.parse_args()
    main(args)

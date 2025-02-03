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
from scipy.io import loadmat
import glob

import sys

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry, recon_volume


def main():
    """Assume CT is in a unit cube. We synthesize X-ray projections."""
    # scanner_cfg_path = 'C:\R2_Gaussian_4D\data_generator\spare_data\cone_beam_cv.yml'
    scanner_cfg_path = 'C:\R2_Gaussian_4D\data_generator\spare_data\cone_beam_mc.yml'
    folder_path = 'Z:\Yabo\SPARE/raw\MonteCarloDatasets'
    file_list = glob.glob(os.path.join(folder_path, "*"), recursive=False)
    file_list = [os.path.basename(f) for f in file_list if os.path.basename(f).startswith('MC')]
    processed_file_list = glob.glob(os.path.join('C:\R2_Gaussian_4D\data/spare_dataset', "*"), recursive=False)
    processed_file_list = [os.path.basename(f) for f in processed_file_list if os.path.basename(f).startswith('MC')]
    for f in file_list:
        case_name = f[:-4]
        if case_name in processed_file_list:
            continue
        # proj_raw_path = 'Z:\Yabo\SPARE/raw\ClinicalVarianDatasets\P2\CV_P2_T_01/CV_P2_T_01.mat'
        proj_raw_path = f'Z:\Yabo\SPARE/raw\MonteCarloDatasets/{case_name}.mat'
        output_path = 'C:\R2_Gaussian_4D\data\spare_dataset'
        case_save_path = osp.join(output_path, case_name)
        os.makedirs(case_save_path, exist_ok=True)

        # Load configuration
        with open(scanner_cfg_path, "r") as handle:
            scanner_cfg = yaml.safe_load(handle)

        print(f"Generate projection data ... ")
        geo = get_geometry(scanner_cfg)

        proj_raw = loadmat(proj_raw_path)
        angles = proj_raw['data']['angles'][0][0].reshape(-1)
        bins = proj_raw['data']['bin'][0][0].reshape(-1)
        real_projs = proj_raw['data']['projs'][0][0]
        real_projs = np.transpose(real_projs, (1, 0, 2))
        # real_projs = 11.08 * real_projs + 139.09
        projs = []
        proj_angles = []

        ttt = []
        for i in range(10):
            t = i/10.
            ii = i+1
            tt = str(ii).zfill(2)
            t_t = str(i*10).zfill(2)
            # vol_path = f"C:\R2_Gaussian_4D\data_generator\spare_data\processed\mc_p1_ns_{tt}.npy"
            # vol = np.load(vol_path).astype(np.float32)
            # vol = vol.transpose([0, 2, 1])
            # vol = np.flip(np.rot90(vol, k=1, axes=(0, 1)), axis=2)
            # vol = np.flip(vol, axis=0)
            # t_angle = angles[bins == ii]
            #
            #
            # np.save(osp.join(case_save_path, f"vol_gt_{t_t}.npy"), vol)

            # Generate training projections
            # projs_train_angles = t_angle / 180 * np.pi

            # projs_train = tigre.Ax(
            #     np.transpose(vol, (2, 1, 0)).copy(), geo, projs_train_angles
            # )[:, ::-1, :]

            # projs.append(projs_train)
            # proj_angles.append(t_angle/180*np.pi)
            # ttt.append(np.ones_like(t_angle)*t)

        # projs = np.concatenate(projs, axis=0)
        # proj_angles = np.concatenate(proj_angles, axis=0)
        # ttt = np.concatenate(ttt, axis=0)
        # sorted_indices = np.argsort(proj_angles)

        # Reorder array_3d based on sorted_indices
        # projs = projs[sorted_indices]
        # projs = projs.transpose([1,2,0])
        # proj_angles = proj_angles[sorted_indices]
        # ttt = ttt[sorted_indices]
        # Save
        os.makedirs(osp.join(case_save_path, 'proj_train'), exist_ok=True)
        os.makedirs(osp.join(case_save_path, 'proj_test'), exist_ok=True)
        file_path = []
        file_path_test = []
        projs = real_projs.astype(np.float32)
        for i_proj in range(projs.shape[2]):
            t = (bins[i_proj] - 1)/10.
            print(f't is {t}')
            proj = projs[:,:,i_proj]
            frame_save_name = osp.join('proj_train', f"train_{i_proj:04d}.npy")
            np.save(osp.join(case_save_path, frame_save_name), proj)
            file_path.append(
                {
                    "file_path": frame_save_name,
                    "angle": float(angles[i_proj]*3.1415926/180),
                    "time": round(float(t),2),
                }
            )
            if i_proj % 20 == 0:
                frame_save_name = osp.join('proj_test', f"test_{i_proj:04d}.npy")
                np.save(osp.join(case_save_path, frame_save_name), proj)
                file_path_test.append(
                    {
                        "file_path": frame_save_name,
                        "angle": float(angles[i_proj]*3.1415926/180),
                        "time": round(float(t), 2),
                    }
                )

        meta = {
            "scanner": scanner_cfg,
            "vol": 'cgls.npy',
            "bbox": [[-1, -1, -1], [1, 1, 1]],
            "proj_train": file_path,
            "proj_test": file_path_test,

        }
        with open(osp.join(case_save_path, "meta_data.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
    print(f"Generate data for case {case_name} complete!")


if __name__ == "__main__":
    # fmt: off
    main()

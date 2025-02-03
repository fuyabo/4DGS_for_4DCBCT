import json
import os
import os.path as osp
import sys

import numpy as np
import yaml
from scipy.io import loadmat

sys.path.append("./")


def main():
    """Assume CT is in a unit cube. We synthesize X-ray projections."""
    case_name = 'xxxx'
    scanner_cfg_path = 'C:\R2_Gaussian_4D\data_generator\hypersight_dataset\cone_beam_cv.yml'
    proj_raw_path = f'Z:\Yabo\CBCT_Recon/{case_name}/data_ten_bins_fx1_sup.mat'
    output_path = 'C:\R2_Gaussian_4D\data\hypersight_dataset'
    case_save_path = osp.join(output_path, case_name)
    os.makedirs(case_save_path, exist_ok=True)

    # Load configuration
    with open(scanner_cfg_path, "r") as handle:
        scanner_cfg = yaml.safe_load(handle)

    proj_raw = loadmat(proj_raw_path)
    angles = proj_raw['data']['angles'][0][0].reshape(-1)
    bins = proj_raw['data']['bin'][0][0].reshape(-1)
    real_projs = proj_raw['data']['projs'][0][0]
    real_projs = np.transpose(real_projs, (1, 0, 2))

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
        # proj = cv2.resize(proj, [1418, 1064], interpolation=cv2.INTER_LINEAR)
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

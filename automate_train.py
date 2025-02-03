import os
import subprocess
import sys
import glob

# Define the parent directory containing the folders
parent_directory = "data/spare_datasets"

# Activate the virtual environment
conda_env_name = 'r2_gaussian'

folder_path = 'Z:\Yabo\SPARE/raw\MonteCarloDatasets'
file_list = glob.glob(os.path.join('C:\R2_Gaussian_4D\data/spare_dataset', "*"), recursive=False)
file_list = [f for f in file_list if os.path.basename(f).startswith('MC')]
output_path = 'Z:\Yabo\SPARE\ReconResults\MonteCarloDatasets\Validation'

for f in file_list:
    fn = os.path.basename(f)
    if os.path.exists(os.path.join(f, 'init_{}.npy'.format(fn))):
        print("init pc exists, next see whether results has been obtained.")
        pt = fn.split('_')[2]
        print('this patient is {}'.format(pt))
        result_folder = os.path.join(output_path, pt, fn)
        if os.path.exists(result_folder) and os.path.isdir(result_folder) and bool(os.listdir(result_folder)):
            print('results for {} is already obtained, skip'.format(result_folder))
        else:
            print('processing {}'.format(fn))
            os.makedirs(result_folder)
            pt_folder_path = os.path.join('data/spare_dataset', fn)
            if os.path.isdir(pt_folder_path):
                # Construct the command to activate the Conda environment and run train.py
                command = f"conda activate {conda_env_name} && python train.py -s {pt_folder_path} -m {result_folder}"
                print(f"Running: {command}")
                # Execute the command in a subprocess
                subprocess.run(command, shell=True, executable="/bin/bash" if os.name != 'nt' else None)


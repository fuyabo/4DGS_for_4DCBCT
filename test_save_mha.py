import os
import SimpleITK as sitk

def modify_offset_in_mha_files_recursive(folder_path, new_offset):
    """
    Recursively modify the offset tag in the header of all `.mha` files in a folder.

    Args:
        folder_path (str): Path to the root folder containing `.mha` files.
        new_offset (list or tuple): New offset values to set, e.g., [x, y, z].
    """
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".mha"):
                file_path = os.path.join(root, file_name)

                # Read the .mha file
                image = sitk.ReadImage(file_path)

                # Set the new offset
                image.SetOrigin(new_offset)

                # Save the modified image back to the file
                sitk.WriteImage(image, file_path)
                print(f"Updated offset for: {file_path}")

# Example usage
folder_path = "Z:\Yabo\SPARE\ReconResults\MonteCarloDatasets\Validation"
new_offset = [-224.5, -109.5, -224.5]  # Replace with your desired offset values
modify_offset_in_mha_files_recursive(folder_path, new_offset)

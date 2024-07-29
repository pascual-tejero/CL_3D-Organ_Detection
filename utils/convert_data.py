from typing import List
import os
import numpy as np
import nibabel as nib


def convert_data(source_folder, output_path):
    """
    Convert from .npy to .nii.gz format
    """
    os.makedirs(output_path, exist_ok=True)
    for root, dirs, files in os.walk(source_folder, topdown=True):
        files_npy = [f for f in files if f.endswith('.npy')]
        for file in files_npy:
            data = np.load(os.path.join(root, file))
            root_rem = root.replace(source_folder, "")
            os.makedirs(os.path.join(output_path, root_rem), exist_ok=True)
            img = nib.Nifti1Image(data, np.eye(4))
            file_name = file.split(".")[0]
            nib.save(img, os.path.join(output_path, root_rem, file_name + ".nii.gz"))

    print(f"Converted .npy files to .nii.gz format and saved to {output_path}")                


# Convert the .npy file to .dcm format
convert_data("./original_datasets/totalsegmentator_TAPv2_160_160_256_CT/", "./dicom_datasets/totalsegmentator_TAPv2_160_160_256_CT/")
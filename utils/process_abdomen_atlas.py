import os
import numpy as np
import nibabel as nib
from pathlib import Path
import random
import torch
import torch.nn.functional as F

ORGAN_TO_LABEL = {
    # "background": 0,
    "liver": 1,
    "kidney_right": 2,
    "spleen": 3,
    "pancreas": 4,
    "kidney_left": 5,
    "stomach": 6,
    "gall_bladder": 7,
    "aorta": 8,
    "postcava": 9,
}

def apply_resizing(data, new_shape, mode_interpolation="trilinear"):
    # Convert data to a tensor and ensure it has shape [D, H, W] by permuting
    data = torch.tensor(data).permute(2, 0, 1)  # Change from [H, W, D] to [D, H, W]
    data = data.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions: [1, 1, D, H, W]

    new_shape = (new_shape[2], new_shape[0], new_shape[1])
    
    # Use "trilinear" interpolation for 3D resizing
    if mode_interpolation == "trilinear":
        data = F.interpolate(data, size=new_shape, mode=mode_interpolation, align_corners=False)
    else:
        data = F.interpolate(data, size=new_shape, mode=mode_interpolation)

    # Remove the batch and channel dimensions, and permute back to [H, W, D]
    data = data.squeeze().permute(1, 2, 0).numpy()  # Convert back to [H, W, D]
    return data

def process_abdomen_atlas(data_dir):
    # Get directories that start with 'BDMAP' and are directories
    bdmap_dirs = [d for d in os.listdir(data_dir) if d.startswith('BDMAP') and 
                  os.path.isdir(os.path.join(data_dir, d))]

    # Shuffle the directories randomly
    random.shuffle(bdmap_dirs)

    # Calculate the split indices for train, validation, and test sets
    train_split = int(len(bdmap_dirs) * 0.7)
    val_split = int(len(bdmap_dirs) * 0.1)

    # Split the directories into train, validation, and test sets
    train_dirs = bdmap_dirs[:train_split]
    val_dirs = bdmap_dirs[train_split:train_split + val_split]
    test_dirs = bdmap_dirs[train_split + val_split:]

    # Move the directories to the corresponding split directories
    for split, dirs in zip(["train", "val", "test"], [train_dirs, val_dirs, test_dirs]):
        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for d in dirs:
            os.rename(os.path.join(data_dir, d), os.path.join(split_dir, d))

    # Process each directory to create segmentation labels and save as .nii.gz and .npy files
    for dirpath, dirnames, filenames in os.walk(script_dir):
        if "segmentations" in dirpath:
            ct_path = Path(dirpath).parent / "ct.nii.gz"
            ct_nifti = nib.load(str(ct_path))
            ct_data = ct_nifti.get_fdata()
            ct_affine = ct_nifti.affine
            ct_header = ct_nifti.header
            label_matrix = np.zeros_like(ct_data, dtype=np.uint8)

            # Create label matrix from organ segmentations
            for filename in filenames:
                if filename.endswith(".nii.gz"):
                    organ_name = filename.split(".")[0]
                    organ_label = ORGAN_TO_LABEL[organ_name]
                    organ_matrix = nib.load(str(Path(dirpath) / filename)).get_fdata()
                    label_matrix[organ_matrix > 0] = organ_label
            label_path = Path(dirpath).parent / "label.nii.gz"
            nib.save(nib.Nifti1Image(label_matrix, ct_affine, ct_header), label_path)

            # Save ct and label as .npy (data.npy and label.npy)
            # np.save(str(Path(dirpath).parent / "data.npy"), ct_data)
            # np.save(str(Path(dirpath).parent / "label.npy"), label_matrix)

    # Resize and crop the data for train, validation, and test sets
    for dirpath, dirnames, filenames in os.walk(data_dir):
        filenames_data_label = [f for f in filenames if f == "ct.nii.gz" or f == "label.nii.gz"]

        if "train" in dirpath and filenames_data_label:
            ct_path = Path(dirpath) / "ct.nii.gz"
            ct_nifti = nib.load(str(ct_path))
            ct_data = ct_nifti.get_fdata()
            ct_affine = ct_nifti.affine
            ct_header = ct_nifti.header

            label_path = Path(dirpath) / "label.nii.gz"
            label_nifti = nib.load(str(label_path))
            label_data = label_nifti.get_fdata()
            label_affine = label_nifti.affine
            label_header = label_nifti.header

            # Determine the new shape for resizing
            if ct_data.shape[2] < 32:
                z_shape = 32 
            elif ct_data.shape[2] >= 32 and ct_data.shape[2] < 256:
                z_shape = (ct_data.shape[2] // 32) * 32
            else:
                z_shape = 256
            xyz_shape = (256, 256, z_shape)
            ct_data = apply_resizing(ct_data, xyz_shape, mode_interpolation="trilinear").astype(np.float32)
            label_data = apply_resizing(label_data, xyz_shape, mode_interpolation="nearest").astype(np.uint8)

            # Crop the data and label to the same shape
            possible_cropping = [160, 192, 224, 256] # Possible cropping sizes of multiple of 32
            crop_shape = random.choice(possible_cropping)
            crop_start_x = random.randint(0, xyz_shape[0] - crop_shape)
            crop_start_y = random.randint(0, xyz_shape[1] - crop_shape)

            ct_data = ct_data[crop_start_x:crop_start_x + crop_shape, crop_start_y:crop_start_y + crop_shape, :]
            label_data = label_data[crop_start_x:crop_start_x + crop_shape, crop_start_y:crop_start_y + crop_shape, :]
            
            # Save resized and cropped nifti files
            ct_path = Path(dirpath) / "ct_resized_cropped.nii.gz"
            nib.save(nib.Nifti1Image(ct_data, ct_affine, ct_header), ct_path)
            label_path = Path(dirpath) / "label_resized_cropped.nii.gz"
            nib.save(nib.Nifti1Image(label_data, label_affine, label_header), label_path)

            # Save resized and cropped data and label as .npy (data_resized_cropped.npy and label_resized_cropped.npy)
            # np.save(str(Path(dirpath) / "ct_resized_cropped.npy"), ct_data)
            # np.save(str(Path(dirpath) / "label_resized_cropped.npy"), label_data)
            np.save(str(Path(dirpath) / "ct.npy"), ct_data)
            np.save(str(Path(dirpath) / "label.npy"), label_data)

        elif ("val" in dirpath or "test" in dirpath) and filenames_data_label:            
            ct_path = Path(dirpath) / "ct.nii.gz"
            ct_nifti = nib.load(str(ct_path))
            ct_data = ct_nifti.get_fdata()
            ct_affine = ct_nifti.affine
            ct_header = ct_nifti.header

            label_path = Path(dirpath) / "label.nii.gz"
            label_nifti = nib.load(str(label_path))
            label_data = label_nifti.get_fdata()
            label_affine = label_nifti.affine
            label_header = label_nifti.header

            # Determine the new shape for resizing
            if ct_data.shape[2] < 32:
                z_shape = 32 
            elif ct_data.shape[2] >= 32 and ct_data.shape[2] < 256:
                z_shape = (ct_data.shape[2] // 32) * 32
            else:
                z_shape = 256
            xyz_shape = (256, 256, z_shape)
            ct_data = apply_resizing(ct_data, xyz_shape, mode_interpolation="trilinear").astype(np.float32)
            label_data = apply_resizing(label_data, xyz_shape, mode_interpolation="nearest").astype(np.uint8)
            
            # Save resized and cropped nifti files
            ct_path = Path(dirpath) / "ct_resized_cropped.nii.gz"
            nib.save(nib.Nifti1Image(ct_data, ct_affine, ct_header), ct_path)
            label_path = Path(dirpath) / "label_resized_cropped.nii.gz"
            nib.save(nib.Nifti1Image(label_data, label_affine, label_header), label_path)

            # Save resized and cropped data and label as .npy (data_resized_cropped.npy and label_resized_cropped.npy)
            # np.save(str(Path(dirpath) / "ct_resized_cropped.npy"), ct_data)
            # np.save(str(Path(dirpath) / "label_resized_cropped.npy"), label_data)
            np.save(str(Path(dirpath) / "ct.npy"), ct_data)
            np.save(str(Path(dirpath) / "label.npy"), label_data)
        
        filenames_data_label = []


if __name__ == "__main__":
    np.random.seed(0)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Processing abdomen atlas data in {script_dir}")
    process_abdomen_atlas(script_dir)
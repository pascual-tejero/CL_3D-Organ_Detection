import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
import os
import shutil

def downsample_data(source_folder, output_folder, size_xyz=64):
    
    os.makedirs(output_folder, exist_ok=True) # Create the output folder if it does not exist
    shutil.copy(os.path.join(source_folder, "data_info.json"), output_folder)    # Get the data_info.json file

    # Go through all the files in the source folder
    for root, dirs, files in os.walk(source_folder, topdown=True):

        files_npy = [f for f in files if f.endswith('.npy')] # Get all .npy files
        for file in files_npy: # For each .npy file
            data = np.load(os.path.join(root, file)) # Load the .npy file

            slice_idx_xy = np.random.randint(0, data.shape[2] - size_xyz) # Randomly select 64 slices in the x-y plane
            slice_idx_z = np.random.randint(0, data.shape[3] - size_xyz)
            # slice_idx_z = 64 # Select the 64th slice in the z plane

            if file == 'data.npy': # If the file is "data.npy"
                data_downsampled = data[:, slice_idx_xy:slice_idx_xy + size_xyz, 
                                        slice_idx_xy:slice_idx_xy + size_xyz, 
                                        slice_idx_z:slice_idx_z + size_xyz]
            elif file == 'label.npy': # If the file is "label.npy"
                data_downsampled = data[:, slice_idx_xy:slice_idx_xy + size_xyz, 
                                        slice_idx_xy:slice_idx_xy + size_xyz, 
                                        slice_idx_z:slice_idx_z + size_xyz]
                
            root_rem = root.replace(source_folder, "") # Remove the source folder from the root
            os.makedirs(os.path.join(output_folder, root_rem), exist_ok=True) # Create the output folder if it does not exist
            np.save(os.path.join(output_folder, root_rem, file), data_downsampled)


def check_downsampled_data(folder, ideal_size=(64,64,64)):

    no_downsampled_data = [] # List to store data that are not padded to ideal shape
    no_downsampeld_label = [] # List to store label that are not padded to ideal shape

    for root, dirs, files in os.walk(folder, topdown=True):
        files_npy = [f for f in files if f.endswith('.npy')] # Get all .npy files
        if files_npy:
            data_downsampled = np.load(os.path.join(root, "data.npy")) # "data.npy" is the padded data
            label_downsampled = np.load(os.path.join(root, "label.npy")) # "label.npy" is the padded label
            # print(f"Data shape: {data_downsampled.shape}")
            # print(f"Label shape: {label_downsampled.shape}")

            if data_downsampled.shape[1:] != ideal_size: # Check if the shape of the padded data is not ideal
                print(f"Data shape is not ideal: {data_downsampled.shape[1:]}")
                print(f"From location: {os.path.join(root, 'data.npy')}")
                no_downsampled_data.append(os.path.join(root, 'data.npy'))
            if label_downsampled.shape[1:] != ideal_size: # Check if the shape of the padded label is not ideal
                print(f"Label shape is not ideal: {label_downsampled.shape[1:]}")
                print(f"From location: {os.path.join(root, 'label.npy')}")
                no_downsampeld_label.append(os.path.join(root, 'label.npy'))

    if no_downsampled_data: # If there are data that are not padded to ideal shape
        print(f"Data that are not downsampled to ideal shape: {no_downsampled_data}")
 
    if no_downsampeld_label: # If there are label that are not padded to ideal shape
        print(f"Label that are not downsampled to ideal shape: {no_downsampeld_label}")

    if not no_downsampled_data and not no_downsampeld_label: # If all data and label are padded to ideal shape
        print(f"All data and label from \"{folder}\" are downsampled to ideal shape {ideal_size}.")

if __name__ == "__main__":

    # source_folder = "./matched_datasets/word_224_224_160_CT/"
    # output_folder = "./downsampled_datasets/word_64_64_64_CT/"
    # downsample_data(source_folder, output_folder, size_xyz=64)
    # check_downsampled_data(output_folder)

    # source_folder = "./original_datasets/totalsegmentator_TAPv2_160_160_256_CT/"
    # output_folder = "./downsampled_datasets/totalsegmentator_TAPv2_160_160_256_CT/"
    # downsample_data(source_folder, output_folder, size_xyz=64)
    # check_downsampled_data(output_folder)

    # source_folder = "./matched_datasets/abdomenCT-1k_224_224_160_CT/"
    # # source_folder = "./padded_datasets/abdomenCT-1k_224_224_160_CT/"
    # output_folder = "./downsampled_datasets/abdomenCT-1k_64_64_64_CT/"
    # downsample_data(source_folder, output_folder, size_xyz=64)
    # check_downsampled_data(output_folder, ideal_size=(64,64,64))

    # check_downsampled_data("./downsampled_datasets/abdomenCT-1k_64_64_64_CT/")
    # check_downsampled_data("./downsampled_datasets/totalsegmentator_TAPv2_160_160_256_CT/")
    # check_downsampled_data("./downsampled_datasets/word_64_64_64_CT/")

    source_folder = "./matched_datasets/total_segmentator_256_256_256_CT/"
    output_folder = "./downsampled_datasets/total_segmentator_64_64_64_CT/"
    downsample_data(source_folder, output_folder, size_xyz=64)
    check_downsampled_data(output_folder, ideal_size=(64,64,64))


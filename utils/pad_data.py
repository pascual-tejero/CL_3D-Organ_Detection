import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
import os
import shutil


def pad_data(source_folder, output_folder, dataset_name):

    if dataset_name == "TOTAL-SEGMENTATOR": # From (:,160,160,256) to (:,224,224,256) 
        pad_size = [64,0]
        folder_name = "total_segmentator_224_224_256_CT"
    elif dataset_name == "WORD": # From (:,224,224,160) to (:,224,224,256) 
        pad_size = [0,96] 
        folder_name = "word_224_224_256_CT"
    elif dataset_name == "ABDOMEN-CT-1K":
        pad_size = [0,64] # From (:,224,224,96) to (:,224,224,160)
        folder_name = "abdomenCT-1k_224_224_160_CT"

    root_output_folder = os.path.join(output_folder, folder_name)
    os.makedirs(root_output_folder, exist_ok=True) # Create the output folder if it does not exist
    shutil.copy(os.path.join(source_folder, "data_info.json"), root_output_folder) # Get the data_info.json file
    
    # Go through all the files in the source folder
    for root, dirs, files in os.walk(source_folder, topdown=True):        

        files_npy = [f for f in files if f.endswith('.npy')] # Get all .npy files
        for file in files_npy: # For each .npy file
            data = np.load(os.path.join(root, file)) # Load the .npy file

            if file == 'data.npy': # If the file is "data.npy"
                data_padded = np.pad(data, ((0,0), (pad_size[0], 0), (0, pad_size[0]),
                                            (0,pad_size[1])), 'constant', constant_values=-1000)
            elif file == 'label.npy': # If the file is "label.npy"
                data_padded = np.pad(data, ((0,0), (pad_size[0], 0), (0, pad_size[0]),
                                            (0, pad_size[1])), 'constant', constant_values=0)

            dir_save = os.path.join(root_output_folder, root.replace(source_folder, "")) # Remove the source folder from the root
            os.makedirs(dir_save, exist_ok=True) # Create the output folder if it does not exist
            np.save(os.path.join(dir_save, file), data_padded) # Save the padded data to the output folder



def check_padded_data(folder, ideal_size=(256,256,256)):

    no_padded_data = [] # List to store data that are not padded to ideal shape
    no_padded_label = [] # List to store label that are not padded to ideal shape

    for root, dirs, files in os.walk(folder, topdown=True):
        files_npy = [f for f in files if f.endswith('.npy')] # Get all .npy files
        if files_npy:
            data_padded = np.load(os.path.join(root, "data.npy")) # "data.npy" is the padded data
            label_padded = np.load(os.path.join(root, "label.npy")) # "label.npy" is the padded label

            if data_padded.shape[1:] != ideal_size: # Check if the shape of the padded data is not ideal
                print(f"Data shape is not ideal: {data_padded.shape[1:]}")
                print(f"From location: {os.path.join(root, 'data.npy')}")
                no_padded_data.append(os.path.join(root, 'data.npy'))
            if label_padded.shape[1:] != ideal_size: # Check if the shape of the padded label is not ideal
                print(f"Label shape is not ideal: {label_padded.shape[1:]}")
                print(f"From location: {os.path.join(root, 'label.npy')}")
                no_padded_label.append(os.path.join(root, 'label.npy'))

    if no_padded_data: # If there are data that are not padded to ideal shape
        print(f"Data that are not padded to ideal shape: {no_padded_data}")
 
    if no_padded_label: # If there are label that are not padded to ideal shape
        print(f"Label that are not padded to ideal shape: {no_padded_label}")

    if not no_padded_data and not no_padded_label: # If all data and label are padded to ideal shape
        print(f"All data and label from {folder} are padded to ideal shape {ideal_size}.")





if __name__ == "__main__":

    source_folder = "totalsegmentator_TAPv2_160_160_256_CT/"
    # source_folder = "word_224_224_160_CT/"
    # source_folder = "abdomenCT-1k_224_224_96_reg_full_CT/"

    source_folder = os.path.join("./original_datasets/", source_folder)
    output_folder = "padded_datasets/"  

    pad_data(source_folder, output_folder, dataset_name="TOTAL-SEGMENTATOR")
    # pad_data(source_folder, output_folder, dataset_name="WORD")
    # pad_data(source_folder, output_folder, dataset_name="ABDOMEN-CT-1K")

    check_padded_data("padded_datasets/total_segmentator_224_224_256_CT", ideal_size=(224,224,256))
    check_padded_data("original_datasets/totalsegmentator_TAPv2_160_160_256_CT", ideal_size=(160,160,256))

    # check_padded_data("padded_datasets/word_224_224_256_CT", ideal_size=(224,224,256))
    # check_padded_data("original_datasets/word_224_224_160_CT", ideal_size=(224,224,160))

    # check_padded_data("padded_datasets/abdomenCT-1k_224_224_160_CT", ideal_size=(224,224,160))
    # check_padded_data("original_datasets/abdomenCT-1k_224_224_96_reg_full_CT", ideal_size=(224,224,96))
        
    pass
import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
import os
import shutil


def check_data_cloned(source_folder, dataset_name="WORD"):
    folder_names_train = []
    folder_names_val = []
    folder_names_test = []
    
    # Go through all the files in the source folder
    for root, dirs, files in os.walk(source_folder, topdown=True):

        # Get for each folder the name of the sample
        for name in dirs:
            if "train" in root:
                folder_names_train.append(name)
            elif "val" in root:
                folder_names_val.append(name)
            elif "test" in root:
                folder_names_test.append(name)

    print(f"Number of samples in train: {len(folder_names_train)}")
    print(f"Number of samples in val: {len(folder_names_val)}")
    print(f"Number of samples in test: {len(folder_names_test)}")

    # Check if there is a sample name that is in any of the other sets
    for folder in folder_names_train:
        if folder in folder_names_val:
            print(f"Folder {folder} is in train and val")
        if folder in folder_names_test:
            print(f"Folder {folder} is in train and test")

    for folder in folder_names_val:
        if folder in folder_names_test:
            print(f"Folder {folder} is in val and test")
        if folder in folder_names_train:
            print(f"Folder {folder} is in val and train")

    for folder in folder_names_test:
        if folder in folder_names_train:
            print(f"Folder {folder} is in test and train")
        if folder in folder_names_val:
            print(f"Folder {folder} is in test and val")

if __name__ == "__main__":
    # source_folder = "./matched_datasets/abdomenCT-1k_224_224_160_CT"
    source_folder = "./matched_datasets/word_224_224_160_CT"
    check_data_cloned(source_folder)
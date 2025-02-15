import os
import shutil
import json
import numpy as np
import copy
import torch
import torch.nn.functional as F


ABDOMENATLAS_MAP_ID_TO_LABEL = {
    0: "background",
    1: "liver",
    2: "kidney_right",
    3: "spleen",
    4: "pancreas",
    5: "kidney_left",
    6: "gallbladder",
    7: "stomach",
    8: "aorta",
    9: "postcava"
}

TOTALSEGMENTATOR_MAP_ID_TO_LABEL = {
    0: "background",
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "pancreas",
    9: "adrenal_gland_right",
    10: "adrenal_gland_left",
    11: "esophagus",
    12: "trachea",
    13: "small_bowel",
    14: "duodenum",
    15: "colon",
    16: "urinary_bladder",
    17: "heart",
    18: "left_lung",
    19: "right_lung"
}

MAP_ABDOMENATLAS_TO_MATCHED_DATASETS = {
    0: 0, # background (0) -> background (0)
    1: 1, # liver (1) -> liver (1)
    2: 2 , # kidney_right (2) -> kidney_right (2)
    3: 3, # spleen (3) -> spleen (3)
    4: 4, # pancreas (4) -> pancreas (4)
    5: 5, # kidney_left (5) -> kidney_left (5)
    6: 6, # gallbladder (6) -> gallbladder (6)
    7: 7, # stomach (7) -> stomach (7)
    8: 8, # aorta (8) -> aorta (8)
    9: 0 # postcava (9) -> background (0)
}

MAP_TOTALSEGMENTATOR_TO_MATCHED_DATASETS = {
    0: 0, # background (0) -> background (0)
    1: 5, # spleen (1) -> liver (5)
    2: 2, # kidney_right (2) -> kidney_right (2)
    3: 1, # kidney_left (3) -> spleen (1)
    4: 8, # gallbladder (4) -> pancreas (8)
    5: 3, # liver (5) -> kidney_left (3)
    6: 4, # stomach (6) -> gallbladder (4)
    7: 6, # aorta (7) -> stomach (6)
    8: 7, # pancreas (8) -> aorta (7)
    9: 9, # adrenal_gland_right (9) -> adrenal_gland_right (9)
    10: 10, # adrenal_gland_left (10) -> esophagus (11)
    11: 11, # esophagus (11) -> esophagus (11)
    12: 12, # trachea (12) -> trachea (12)
    13: 13, # small_bowel (13) -> small_bowel (13)
    14: 14, # duodenum (14) -> duodenum (14)
    15: 15, # colon (15) -> colon (15)
    16: 16, # urinary_bladder (16) -> urinary_bladder (16)
    17: 17, # heart (17) -> heart (17)
    18: 18, # left_lung (18) -> left_lung (18)
    19: 19 # right_lung (19) -> right_lung (19)
}

MATCHED_DATASETS_MAP_ID_TO_LABEL = {
    0: "background",
    1: "liver",
    2: "kidney_right",
    3: "spleen",
    4: "pancreas",
    5: "kidney_left",
    6: "gallbladder",
    7: "stomach",
    8: "aorta",
    9: "adrenal_gland_right",
    10: "adrenal_gland_left",
    11: "esophagus",
    12: "trachea",
    13: "small_bowel",
    14: "duodenum",
    15: "colon",
    16: "urinary_bladder",
    17: "heart",
    18: "left_lung",
    19: "right_lung"
}


def check_match_labels_datasets_json(datasets_path_list):
    common_labels = False

    for dataset_path in datasets_path_list:
        with open(os.path.join(dataset_path, "data_info.json")) as f:
            dataset_info = json.load(f)

        dataset_labels = dataset_info["labels"]
        if not common_labels:
            common_labels = set(dataset_labels.values())
            diff_labels = set(dataset_labels.values())  
        else:
            common_labels = set(dataset_labels.values()) & set(common_labels)
            diff_labels = set(dataset_labels.values()) ^ set(diff_labels)

    print(f"Common labels ({len(common_labels)}): {common_labels}")
    print(f"Uncommon labels ({len(diff_labels)}): {diff_labels}")

def check_all_labels_dataset(source_folder):
    unique_labels = set()   

    # Go through all the files in the source folder
    for root, dirs, files in os.walk(source_folder, topdown=True):
        files_npy = [f for f in files if f.endswith('.npy')] # Get all .npy files

        for file in files_npy: # For each .npy file
            if file == "label.npy":
                label = np.load(os.path.join(root, file))
                unique_labels.update(np.unique(label))

    print(f"Unique labels from {source_folder}: {unique_labels}")

def analyse_labels_dataset(source_folder, dataset_name):
    if dataset_name == "ABDOMENATLAS":
        dict_count = {f"{value} ({key})": 0 for key, value in ABDOMENATLAS_MAP_ID_TO_LABEL.items()}
        dict_mean = {f"{value} ({key})": 0 for key, value in ABDOMENATLAS_MAP_ID_TO_LABEL.items()}
    elif dataset_name == "TOTALSEGMENTATOR":
        dict_count = {f"{value} ({key})": 0 for key, value in TOTALSEGMENTATOR_MAP_ID_TO_LABEL.items()}
        dict_mean = {f"{value} ({key})": 0 for key, value in TOTALSEGMENTATOR_MAP_ID_TO_LABEL.items()}
    elif dataset_name == "MATCHED_DATASETS":
        dict_count = {f"{value} ({key})": 0 for key, value in MATCHED_DATASETS_MAP_ID_TO_LABEL.items()}
        dict_mean = {f"{value} ({key})": 0 for key, value in MATCHED_DATASETS_MAP_ID_TO_LABEL.items()}

    # Go through all the files in the source folder
    count_data = 0
    count_scans = 0
    for root, dirs, files in os.walk(source_folder, topdown=True):
        files_npy = [f for f in files if f.endswith('.npy')] # Get all .npy files

        for file in files_npy: # For each .npy file
            if file == "label.npy":
                label = np.load(os.path.join(root, file))
                count_data += 1
                unique_labels, counts = np.unique(label, return_counts=True)
                for i in range(len(unique_labels)):
                    label_unique = unique_labels[i]
                    if dataset_name == "ABDOMENATLAS":
                        label_name = ABDOMENATLAS_MAP_ID_TO_LABEL[label_unique]
                        dict_count[f"{label_name} ({int(label_unique)})"] += counts[i]
                    elif dataset_name == "TOTALSEGMENTATOR":
                        label_name = TOTALSEGMENTATOR_MAP_ID_TO_LABEL[label_unique]
                        dict_count[f"{label_name} ({int(label_unique)})"] += counts[i]
                    elif dataset_name == "MATCHED_DATASETS":
                        label_name = MATCHED_DATASETS_MAP_ID_TO_LABEL[label_unique]
                        dict_count[f"{label_name} ({int(label_unique)})"] += counts[i]
            elif file == "data.npy":
                count_scans += 1
    if count_data == 0:
        count_data = 1
    dict_mean = {key: round(value / count_data, 2) for key, value in dict_count.items()}
    print(f"Count of labels from \"{source_folder}\":\n {dict_count}")
    print(f"Mean of labels from \"{source_folder}\":\n {dict_mean}")
    print(f"Number of scans from \"{source_folder}\": {count_scans}")
    print("===============================================")


def match_datasets(source_folder, matched_dataset_folder, dataset_name):

    if dataset_name == "TOTALSEGMENTATOR":
        folder_name = "total_segmentator_256_256_256_CT"
    elif dataset_name == "ABDOMENATLAS":
        folder_name = "abdomen_atlas_256_256_256_CT"

    root_output_folder = os.path.join(matched_dataset_folder, folder_name)
    os.makedirs(root_output_folder, exist_ok=True) # Create the output folder if it does not exist

    # Get the data_info.json file
    shutil.copy(os.path.join(source_folder, "data_info.json"), root_output_folder)

    data_info = {}
    if dataset_name == "TOTALSEGMENTATOR":
        data_info["num_classes"] = 19
        dict_copy = copy.deepcopy(MATCHED_DATASETS_MAP_ID_TO_LABEL)
        del dict_copy[0]
        data_info["labels"] = dict_copy
        data_info["labels_small"] = {4: 'pancreas', 6: 'gallbladder',  9: 'adrenal_gland_right', 10: 'adrenal_gland_left',
                                     11: 'esophagus',  12: 'trachea', 16: 'urinary_bladder'}
        data_info["labels_mid"] = {2: 'kidney_right', 3: 'spleen', 5: 'kidney_left', 8: 'aorta', 14: 'duodenum', }
        data_info["labels_large"] = {1: 'liver', 7: 'stomach', 15: 'colon', 13: 'small_bowel', 17: 'heart', 
                                     18: 'left_lung', 19: 'right_lung'}

    elif dataset_name == "ABDOMENATLAS":
        data_info["num_classes"] = 8
        dict_copy = copy.deepcopy(MATCHED_DATASETS_MAP_ID_TO_LABEL)
        del dict_copy[0], dict_copy[9], dict_copy[10], dict_copy[11], dict_copy[12], dict_copy[13], dict_copy[14], dict_copy[15], dict_copy[16], dict_copy[17], dict_copy[18], dict_copy[19]
        data_info["labels"] = dict_copy
        data_info["labels_small"] = {2: "right kidney", 3: "spleen", 5: "left kidney", 6: "stoamch"}
        data_info["labels_mid"] = {4: "pancreas", 7: "gallbladder", 8: "aorta", 9: "aorta"}
        data_info["labels_large"] = {1: "liver"}

    with open(os.path.join(root_output_folder, "data_info.json"), 'w') as f:
        json.dump(data_info, f, indent=4)

    # Go through all the files in the source folder
    for root, dirs, files in os.walk(source_folder, topdown=True):

        files_npy = [f for f in files if f.endswith('.npy')] # Get all .npy files

        if files_npy: # If there are .npy files
            print(f"Processing {root}")
            data = np.load(os.path.join(root, "data.npy")) # "data.npy" is the original data
            label = np.load(os.path.join(root, "label.npy")) # "label.npy" is the original label

            label_matched = np.zeros(label.shape, dtype=np.uint8)

            if dataset_name == "ABDOMENATLAS": 
                for key, value in MAP_ABDOMENATLAS_TO_MATCHED_DATASETS.items():
                    label_matched[label == key] = value
            elif dataset_name == "TOTALSEGMENTATOR": 
                # Interpolate data to size 256x256x256
                data = torch.tensor(data).unsqueeze(0).float() # (1, 160, 160, 256)
                data = F.interpolate(data, size=(256, 256, 256), mode='trilinear', align_corners=False).squeeze(0).numpy() # (1, 256, 256, 256)

                for key, value in MAP_TOTALSEGMENTATOR_TO_MATCHED_DATASETS.items():
                    label_matched[label == key] = value

            label_matched = torch.tensor(label_matched).unsqueeze(0).float() # (1, 160, 160, 256)
            label_matched = F.interpolate(label_matched, size=(256, 256, 256), mode='nearest').squeeze(0).numpy() # (1, 256, 256, 256)

            # Save the matched data and label
            dir_save = os.path.join(root_output_folder, root.replace(source_folder, "").lstrip("/\\"))

            os.makedirs(dir_save, exist_ok=True) # Create the output folder if it does not exist
            np.save(os.path.join(dir_save, "data.npy"), data) # Save the matched data to the matched_dataset_folder
            np.save(os.path.join(dir_save, "label.npy"), label_matched) # Save the matched label to the matched_dataset_folder
            


if __name__ == "__main__":
    pass

    # Make matched dataset for TOTALSEGMENTATOR
    match_datasets(source_folder="./totalsegmentator_TAPv2_160_160_256_CT", matched_dataset_folder="./matched_datasets/", dataset_name="TOTALSEGMENTATOR")
    analyse_labels_dataset(source_folder="./totalsegmentator_TAPv2_160_160_256_CT", dataset_name="TOTALSEGMENTATOR")
    analyse_labels_dataset(source_folder="./matched_datasets/total_segmentator_256_256_256_CT/", dataset_name="MATCHED_DATASETS")

    # Make matched dataset for ABDOMENATLAS
    match_datasets(source_folder="./AbdomenAtlas", matched_dataset_folder="./matched_datasets/", dataset_name="ABDOMENATLAS")
    analyse_labels_dataset(source_folder="./AbdomenAtlas", dataset_name="ABDOMENATLAS")
    analyse_labels_dataset(source_folder="./matched_datasets/abdomen_atlas_256_256_256_CT/", dataset_name="MATCHED_DATASETS")

    # match_datasets(source_folder="./totalsegmentator_TAPv2_160_160_256_CT", matched_dataset_folder="./matched_datasets/", dataset_name="TOTALSEGMENTATOR")
    # analyse_labels_dataset(source_folder="./totalsegmentator_TAPv2_160_160_256_CT", dataset_name="TOTALSEGMENTATOR")
    # analyse_labels_dataset(source_folder="./matched_datasets/total_segmentator_256_256_256_CT/", dataset_name="MATCHED_DATASETS")

    # match_datasets(source_folder="./AbdomenAtlas_processed", matched_dataset_folder="./matched_datasets/", dataset_name="ABDOMENATLAS")
    # analyse_labels_dataset(source_folder="./AbdomenAtlas_processed", dataset_name="ABDOMENATLAS")
    # analyse_labels_dataset(source_folder="./matched_datasets/abdomen_atlas_256_256_256_CT/", dataset_name="MATCHED_DATASETS")
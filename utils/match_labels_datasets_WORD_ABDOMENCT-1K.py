import os
import shutil
import json
import numpy as np
import copy

WORD_MAP_ID_TO_LABEL = {
    0: "background (0)",
    1: "liver (1)",
    2: "spleen (2)",
    3: "left kidney (3)",
    4: "right kidney (4)",
    5: "stomach (5)",
    6: "gallbladder (6)",
    7: "pancreas (7)",
    8: "duodenum (8)",
    9: "colon (9)",
    10: "intestine (10)",
    11: "left adrenal gland (11)",
    12: "rectum (12)",
    13: "bladder (13)",
    14: "right adrenal gland (14)"
}

ABDOMENCT1K_MAP_ID_TO_LABEL = {
    0: "background",
    1: "liver",
    2: "right kidney",
    3: "spleen",
    4: "pancreas",
    5: "left kidney",
}

ABDOMENCT1K_MAP_ID_TO_LABEL_WITH_ID = {
    0: "background (0)",
    1: "liver (1)",
    2: "right kidney (2)",
    3: "spleen (3)",
    4: "pancreas (4)",
    5: "left kidney (5)",
}

MAP_WORD_TO_MATCHED_DATASETS = {
    0: 0, # background (0) -> background (0)
    1: 1, # liver (1) -> liver (1)
    2: 3, # spleen (2) -> spleen (3)
    3: 5, # left kidney (3) -> left kidney (5)
    4: 2, # right kidney (4) -> right kidney (2)
    5: 6, # stomach (5) -> stomach (6)
    6: 0, # gallbladder (6) -> background (0)
    7: 4, # pancreas (7) -> pancreas (4)
    8: 7, # duodenum (8) -> duodenum (7)
    9: 8, # colon (9) -> colon (8)
    10: 9, # intestine (10) -> intestine (9)
    11: 0, # left adrenal gland (11) -> background (0)
    12: 8, # rectum (12) -> colon (8)
    13: 10, # bladder (13) -> bladder (10)
    14: 0 # right adrenal gland (14) -> background (0)
}

MATCHED_DATASETS_MAP_ID_TO_LABEL = {
    1: "liver",
    2: "right kidney",
    3: "spleen",
    4: "pancreas",
    5: "left kidney",
    6: "stomach",
    7: "duodenum",
    8: "colon",
    9: "intestine",
    10: "bladder"
}


MATCHED_DATASETS_MAP_ID_TO_LABEL_WITH_ID = {
    0: "background (0)",
    1: "liver (1)",
    2: "right kidney (2)",
    3: "spleen (3)",
    4: "pancreas (4)",
    5: "left kidney (5)",
    6: "stomach (6)",
    7: "duodenum (7)",
    8: "colon (8)",
    9: "intestine (9)",
    10: "bladder (10)"
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
    if dataset_name == "WORD":
        dict_count = {value: 0 for value in WORD_MAP_ID_TO_LABEL.values()}
        dict_mean = {value: 0 for value in WORD_MAP_ID_TO_LABEL.values()}
    elif dataset_name == "ABDOMEN-CT-1K":
        dict_count = {value: 0 for value in ABDOMENCT1K_MAP_ID_TO_LABEL_WITH_ID.values()}
        dict_mean = {value: 0 for value in ABDOMENCT1K_MAP_ID_TO_LABEL_WITH_ID.values()}
    elif dataset_name == "MATCHED_DATASETS":
        dict_count = {value: 0 for value in MATCHED_DATASETS_MAP_ID_TO_LABEL_WITH_ID.values()}
        dict_mean = {value: 0 for value in MATCHED_DATASETS_MAP_ID_TO_LABEL_WITH_ID.values()}

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
                    if dataset_name == "WORD":
                        label_name = WORD_MAP_ID_TO_LABEL[unique_labels[i]]
                        dict_count[label_name] += counts[i]
                    elif dataset_name == "ABDOMEN-CT-1K":
                        label_name = ABDOMENCT1K_MAP_ID_TO_LABEL_WITH_ID[unique_labels[i]]
                        dict_count[label_name] += counts[i]
                    elif dataset_name == "MATCHED_DATASETS":
                        label_name = MATCHED_DATASETS_MAP_ID_TO_LABEL_WITH_ID[unique_labels[i]]
                        dict_count[label_name] += counts[i]
            elif file == "data.npy":
                count_scans += 1
    if count_data == 0:
        count_data = 1
    dict_mean = {key: round(value / count_data, 2) for key, value in dict_count.items()}
    print(f"Count of labels from \"{source_folder}\":\n {dict_count}")
    print(f"Mean of labels from \"{source_folder}\":\n {dict_mean}")
    print(f"Number of scans from \"{source_folder}\": {count_scans}")
    print("===============================================")


def make_matched_dataset(source_folder, matched_dataset_folder, dataset_name):

    if dataset_name == "WORD":
        folder_name = "word_224_224_160_CT"
    elif dataset_name == "ABDOMEN-CT-1K":
        folder_name = "abdomenCT-1k_224_224_160_CT"

    root_output_folder = os.path.join(matched_dataset_folder, folder_name)
    os.makedirs(root_output_folder, exist_ok=True) # Create the output folder if it does not exist

    # Get the data_info.json file
    shutil.copy(os.path.join(source_folder, "data_info.json"), root_output_folder)

    # Read data_info.json and change "num_classes" to 19  and "labels" to MAP_MATCHED_LABELS_ID_TO_LABEL
    with open(os.path.join(root_output_folder, "data_info.json"), 'r') as f:
        data_info = json.load(f)
        data_info["num_classes"] = 14
        data_info["labels"] = MATCHED_DATASETS_MAP_ID_TO_LABEL

    if dataset_name == "ABDOMEN-CT-1K":
        data_info["num_classes"] = 5
        dict_copy = copy.deepcopy(ABDOMENCT1K_MAP_ID_TO_LABEL)
        del dict_copy[0]
        data_info["labels"] = dict_copy
        data_info["labels_small"] = {4: "pancreas"}
        data_info["labels_mid"] = {2: "right kidney", 3: "spleen", 5: "left kidney"}
        data_info["labels_large"] = {1: "liver"}

    elif dataset_name == "WORD":
        data_info["num_classes"] = 10
        dict_copy = MATCHED_DATASETS_MAP_ID_TO_LABEL
        data_info["labels"] = dict_copy
        data_info["labels_small"] = {4: "pancreas", 7: "duodenum"}
        data_info["labels_mid"] = {2: "right kidney", 3: "spleen", 5: "left kidney", 10: "bladder"}
        data_info["labels_large"] = {1: "liver", 6: "stomach", 8: "colon", 9: "intestine"}

    with open(os.path.join(root_output_folder, "data_info.json"), 'w') as f:
        json.dump(data_info, f, indent=4)

    # Go through all the files in the source folder
    for root, dirs, files in os.walk(source_folder, topdown=True):

        files_npy = [f for f in files if f.endswith('.npy')] # Get all .npy files

        if files_npy: # If there are .npy files
            data = np.load(os.path.join(root, "data.npy")) # "data.npy" is the original data
            label = np.load(os.path.join(root, "label.npy")) # "label.npy" is the original label

            label_matched = np.zeros(label.shape, dtype=np.uint8)

            if dataset_name == "WORD": # If the dataset is WORD, we need to change the label 
                for key, value in MAP_WORD_TO_MATCHED_DATASETS.items():
                    label_matched[label == key] = value
            elif dataset_name == "ABDOMEN-CT-1K": # If the dataset is ABDOMENCT1K, we don't need to change the label
                label_matched = label

            # Save the matched data and label
            dir_save = os.path.join(root_output_folder, root.replace(source_folder, "")) # Remove the source folder from the root

            os.makedirs(dir_save, exist_ok=True) # Create the output folder if it does not exist
            np.save(os.path.join(dir_save, "data.npy"), data) # Save the matched data to the matched_dataset_folder
            np.save(os.path.join(dir_save, "label.npy"), label_matched) # Save the matched label to the matched_dataset_folder
            


if __name__ == "__main__":

    # Make matched dataset for WORD
    make_matched_dataset("./original_datasets/word_224_224_160_CT/", "./matched_datasets/", "WORD")
    analyse_labels_dataset("./original_datasets/word_224_224_160_CT/", "WORD")
    analyse_labels_dataset("./matched_datasets/word_224_224_160_CT/", "MATCHED_DATASETS")

    # Make matched dataset for ABDOMEN-CT-1K
    make_matched_dataset("./padded_datasets/abdomenCT-1k_224_224_160_CT/", "./matched_datasets/", "ABDOMEN-CT-1K")
    analyse_labels_dataset("./padded_datasets/abdomenCT-1k_224_224_160_CT/", "ABDOMEN-CT-1K")
    analyse_labels_dataset("./matched_datasets/abdomenCT-1k_224_224_160_CT/", "MATCHED_DATASETS")

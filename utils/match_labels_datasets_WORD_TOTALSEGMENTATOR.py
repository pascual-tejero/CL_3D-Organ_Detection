import os
import shutil
import json
import numpy as np
import copy

WORD_MAP_ID_TO_LABEL = {
    0: "background",
    1: "liver",
    2: "spleen",
    3: "left kidney",
    4: "right kidney",
    5: "stomach",
    6: "gallbladder",
    7: "pancreas",
    8: "duodenum",
    9: "colon",
    10: "intestine",
    11: "left adrenal gland",
    12: "rectum",
    13: "bladder",
    14: "right adrenal gland"
}

WORD_MAP_ID_TO_LABEL_WITH_ID = {
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

TOTALSEGMENTATOR_MAP_ID_TO_LABEL = {
    0: "background",
    1: "liver",
    2: "kidney_right",
    3: "kidney_left",
    4: "pancreas",
    5: "spleen",
    6: "stomach",
    7: "duodenum",
    8: "colon",
    9: "intestine",
    10: "bladder",
    11: "gallbladder",
    12: "esophagus",
    13: "left adrenal gland",
    14: "right adrenal gland",
    15: "trachea",
    16: "aorta",
    17: "heart",
    18: "left lung",
    19: "right lung"
}

TOTALSEGMENTATOR_MAP_ID_TO_LABEL_WITH_ID = {
      0: "background (0)",
      1: "spleen (1)",
      2: "kidney_right (2)",
      3: "kidney_left (3)",
      4: "gallbladder (4)",
      5: "liver (5)",
      6: "stomach (6)",
      7: "aorta (7)",
      8: "pancreas (8)",
      9: "adrenal_gland_right (9)",
      10: "adrenal_gland_left (10)",
      11: "esophagus (11)",
      12: "trachea (12)",
      13: "small_bowel (13)",
      14: "duodenum (14)",
      15: "colon (15)",
      16: "urinary_bladder (16)",
      17: "heart (17)",
      18: "left_lung (18)",
      19: "right_lung (19)",
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

MAP_TOTALSEGMENTATOR_TO_MATCHED_DATASETS = {
    0: 0, # background (0) -> background (0)
    1: 3, # spleen (1) -> spleen (3)
    2: 2, # kidney_right (2) -> kidney_right (2)
    3: 5, # kidney_left (3) -> kidney_left (5)
    4: 11, # gallbladder (4) -> gallbladder (11)
    5: 1, # liver (5) -> liver (1)
    6: 6, # stomach (6) -> stomach (6)
    7: 16, # aorta (7) -> aorta (16)
    8: 4, # pancreas (8) -> pancreas (4)
    9: 14, # adrenal_gland_right (9) -> adrenal_gland_right (14)
    10: 13, # adrenal_gland_left (10) -> adrenal_gland_left (13)
    11: 12, # esophagus (11) -> esophagus (12)
    12: 15, # trachea (12) -> trachea (15)
    13: 9, # small_bowel (13) -> small_bowel, intestine (9)
    14: 7, # duodenum (14) -> duodenum (7)
    15: 8, # colon (15) -> colon (8)
    16: 10, # urinary_bladder (16) -> urinary_bladder, bladder (10)
    17: 17, # heart (17) -> heart (17)
    18: 18, # left_lung (18) -> left_lung (18)
    19: 19, # right_lung (19) -> right_lung (19)
}

MATCHED_DATASETS_MAP_ID_TO_LABEL = {
    0: "background",
    1: "liver",
    2: "right kidney",
    3: "spleen",
    4: "pancreas",
    5: "left kidney",
    6: "stomach",
    7: "duodenum",
    8: "colon",
    9: "intestine",
    10: "bladder",
    11: "gallbladder",
    12: "esophagus",
    13: "left adrenal gland",
    14: "right adrenal gland",
    15: "trachea",
    16: "aorta",
    17: "heart",
    18: "left lung",
    19: "right lung",
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
    10: "bladder (10)",
    11: "gallbladder (11)",
    12: "esophagus (12)",
    13: "left adrenal gland (13)",
    14: "right adrenal gland (14)",
    15: "trachea (15)",
    16: "aorta (16)",
    17: "heart (17)",
    18: "left lung (18)",
    19: "right lung (19)",
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
        dict_count = {value: 0 for value in WORD_MAP_ID_TO_LABEL_WITH_ID.values()}
        dict_mean = {value: 0 for value in WORD_MAP_ID_TO_LABEL_WITH_ID.values()}
    elif dataset_name == "TOTALSEGMENTATOR":
        dict_count = {value: 0 for value in TOTALSEGMENTATOR_MAP_ID_TO_LABEL_WITH_ID.values()}
        dict_mean = {value: 0 for value in TOTALSEGMENTATOR_MAP_ID_TO_LABEL_WITH_ID.values()}
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
                        label_name = WORD_MAP_ID_TO_LABEL_WITH_ID[unique_labels[i]]
                        dict_count[label_name] += counts[i]
                    elif dataset_name == "TOTALSEGMENTATOR":
                        label_name = TOTALSEGMENTATOR_MAP_ID_TO_LABEL_WITH_ID[unique_labels[i]]
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

    if dataset_name == "TOTALSEGMENTATOR":
        folder_name = "total_segmentator_224_224_256_CT"
    elif dataset_name == "WORD":
        folder_name = "word_224_224_256_CT"

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
        data_info["labels_small"] = {4: 'pancreas', 10: 'urinary_bladder', 11:"gallbladder", 12: 'esophagus', 
                                     13: 'adrenal_gland_left', 14: 'adrenal_gland_right', 15: 'trachea'}
        data_info["labels_mid"] = {2: 'kidney_right', 3: 'spleen',  5: 'kidney_left', 7: 'duodenum', 16: 'aorta'}
        data_info["labels_large"] = {1: 'liver', 6: 'stomach', 8: 'colon', 9: 'intestine', 17: 'heart', 
                                     18: 'left_lung', 19: 'right_lung'}

    elif dataset_name == "WORD":
        data_info["num_classes"] = 10
        dict_copy = copy.deepcopy(MATCHED_DATASETS_MAP_ID_TO_LABEL)
        del dict_copy[0], dict_copy[11], dict_copy[12], dict_copy[13], dict_copy[14], dict_copy[15], \
            dict_copy[16], dict_copy[17], dict_copy[18], dict_copy[19]
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
            elif dataset_name == "TOTALSEGMENTATOR": # If the dataset is ABDOMENCT1K, we don't need to change the label
                for key, value in MAP_TOTALSEGMENTATOR_TO_MATCHED_DATASETS.items():
                    label_matched[label == key] = value

            # Save the matched data and label
            dir_save = os.path.join(root_output_folder, root.replace(source_folder, "")) # Remove the source folder from the root

            os.makedirs(dir_save, exist_ok=True) # Create the output folder if it does not exist
            np.save(os.path.join(dir_save, "data.npy"), data) # Save the matched data to the matched_dataset_folder
            np.save(os.path.join(dir_save, "label.npy"), label_matched) # Save the matched label to the matched_dataset_folder
            


if __name__ == "__main__":
    pass

    # Make matched dataset for TOTALSEGMENTATOR
    # make_matched_dataset("./padded_datasets/total_segmentator_224_224_256_CT/", "./matched_datasets/", "TOTALSEGMENTATOR")
    # analyse_labels_dataset("./padded_datasets/total_segmentator_224_224_256_CT/", "TOTALSEGMENTATOR")
    # analyse_labels_dataset("./matched_datasets/total_segmentator_224_224_256_CT/", "MATCHED_DATASETS")

    # Make matched dataset for WORD
    make_matched_dataset("./padded_datasets/word_224_224_256_CT/", "./matched_datasets/", "WORD")
    analyse_labels_dataset("./padded_datasets/word_224_224_256_CT/", "WORD")
    analyse_labels_dataset("./matched_datasets/word_224_224_256_CT/", "MATCHED_DATASETS")


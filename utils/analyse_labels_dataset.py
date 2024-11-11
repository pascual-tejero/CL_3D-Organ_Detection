
import os
import numpy as np

ABDOMEN_ATLAS_MAP_ID_TO_LABEL = {
    "background": 0,
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

ABDOMEN_ATLAS_MAP_LABEL_TO_ID = {
    0: "background",
    1: "liver",
    2: "kidney_right",
    3: "spleen",
    4: "pancreas",
    5: "kidney_left",
    6: "stomach",
    7: "gall_bladder",
    8: "aorta",
    9: "postcava",
}

def analyse_labels_dataset(source_folder, dataset_name):
    # if dataset_name == "WORD":
    #     dict_count = {value: 0 for value in WORD_MAP_ID_TO_LABEL.values()}
    #     dict_mean = {value: 0 for value in WORD_MAP_ID_TO_LABEL.values()}
    # elif dataset_name == "ABDOMEN-CT-1K":
    #     dict_count = {value: 0 for value in ABDOMENCT1K_MAP_ID_TO_LABEL_WITH_ID.values()}
    #     dict_mean = {value: 0 for value in ABDOMENCT1K_MAP_ID_TO_LABEL_WITH_ID.values()}
    # elif dataset_name == "MATCHED_DATASETS":
    #     dict_count = {value: 0 for value in MATCHED_DATASETS_MAP_ID_TO_LABEL_WITH_ID.values()}
    #     dict_mean = {value: 0 for value in MATCHED_DATASETS_MAP_ID_TO_LABEL_WITH_ID.values()}
    if dataset_name == "ABDOMEN_ATLAS":
        dict_count = {value: 0 for value in ABDOMEN_ATLAS_MAP_ID_TO_LABEL.keys()}
        dict_mean = {value: 0 for value in ABDOMEN_ATLAS_MAP_ID_TO_LABEL.keys()}

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
                    if dataset_name == "ABDOMEN_ATLAS":
                        label_name = ABDOMEN_ATLAS_MAP_LABEL_TO_ID[unique_labels[i]]
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

if __name__ == "__main__":
    source_folder = "C:\\Users\\pascu\\Desktop\\master_thesis\\project_pascual\\AbdomenAtlas"
    analyse_labels_dataset(source_folder, "ABDOMEN_ATLAS")
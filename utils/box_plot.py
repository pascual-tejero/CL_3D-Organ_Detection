import numpy as np
import os 
import matplotlib.pyplot as plt

MATCHED_DATASETS_MAP_ID_TO_LABEL_WORD160_ABDOMENCT1K = {
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

MATCHED_DATASETS_MAP_ID_TO_LABEL_WORD256_TOTALSEGMENTATOR = {
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

def box_plot_dataset(dataset_path, dataset_name):

    if dataset_name == "WORD160":
        dict_relative_volume = {value: [] for value in MATCHED_DATASETS_MAP_ID_TO_LABEL_WORD160_ABDOMENCT1K.keys()}
    elif dataset_name == "ABDOMENCT-1K":
        dict_relative_volume = {value: [] for value in MATCHED_DATASETS_MAP_ID_TO_LABEL_WORD160_ABDOMENCT1K.keys()}
        dict_relative_volume = {key: dict_relative_volume[key] for key in range(1, 6)} # Get only the first 5 labels
    elif dataset_name == "WORD256": 
        dict_relative_volume = {value: [] for value in MATCHED_DATASETS_MAP_ID_TO_LABEL_WORD256_TOTALSEGMENTATOR.keys()}
        dict_relative_volume = {key: dict_relative_volume[key] for key in range(1, 11)} # Get only the first 11 labels
    elif dataset_name == "TOTALSEGMENTATOR":
        dict_relative_volume = {value: [] for value in MATCHED_DATASETS_MAP_ID_TO_LABEL_WORD256_TOTALSEGMENTATOR.keys()}
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    
    
    for root, dirs, files in os.walk(dataset_path, topdown=True):
        files_npy = [f for f in files if f.endswith('.npy')] # Get all .npy files

        for file in files_npy: # For each .npy file
            if file == "label.npy":
                label = np.load(os.path.join(root, file))
                unique_labels, counts = np.unique(label, return_counts=True)
                total_volume = np.prod(label.shape)

                count = 1 # Do not count the background

                for i in range(1, len(dict_relative_volume) + 1):
                    if i not in unique_labels:
                        dict_relative_volume[i].append(0)
                    else:
                        dict_relative_volume[i].append(counts[count] / total_volume)
                        count += 1
    
    box_plot(dict_relative_volume, dataset_name)

    return dict_relative_volume

def box_plot(dict_relative_volume, dataset_name):
    fig, ax = plt.subplots()
    ax.boxplot(dict_relative_volume.values())

    if dataset_name == "WORD160":
        labels = list(MATCHED_DATASETS_MAP_ID_TO_LABEL_WORD160_ABDOMENCT1K.values())
    elif dataset_name == "ABDOMENCT-1K":
        labels = list(MATCHED_DATASETS_MAP_ID_TO_LABEL_WORD160_ABDOMENCT1K.values())[:5]
    elif dataset_name == "WORD256":
        labels = list(MATCHED_DATASETS_MAP_ID_TO_LABEL_WORD256_TOTALSEGMENTATOR.values())[:10]
    elif dataset_name == "TOTALSEGMENTATOR":
        labels = list(MATCHED_DATASETS_MAP_ID_TO_LABEL_WORD256_TOTALSEGMENTATOR.values())

    list_number = list(range(1, len(labels) + 1))
    ax.set_xticklabels(list_number, size="large")
    ax.set_yticklabels(ax.get_yticks().round(2), size="large")

    plt.show()

if __name__ == "__main__":
    # dataset_path = "./matched_datasets/abdomenCT-1k_224_224_160_CT/"
    # dataset_name = "ABDOMENCT-1K"    

    # dataset_path = "./matched_datasets/word_224_224_160_CT/"
    # dataset_name = "WORD160" 

    # dataset_path = "./matched_datasets/word_224_224_256_CT/"
    # dataset_name = "WORD256"

    dataset_path = "./matched_datasets/total_segmentator_224_224_256_CT/"
    dataset_name = "TOTALSEGMENTATOR"

    relative_volume = box_plot_dataset(dataset_path, dataset_name)


    for key, value in relative_volume.items():
        print(key, len(value))

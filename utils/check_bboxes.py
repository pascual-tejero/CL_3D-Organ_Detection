import os
import shutil
import json
import numpy as np
from skimage.measure import regionprops

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

TOTAL_SEGMENTATOR_MAP_ID_TO_LABEL = {
    0: "background (0)",
    1: "spleen (1)",
    2: "right kidney (2)", # kidney_right
    3: "left kidney (3)", # kidney_left
    4: "gallbladder (4)",
    5: "liver (5)",
    6: "stomach (6)",
    7: "aorta (7)",
    8: "pancreas (8)",
    9: "right adrenal gland (9)", # adrenal_gland_right
    10: "left adrenal gland (10)", # adrenal_gland_left
    11: "esophagus (11)",
    12: "trachea (12)",
    13: "intestine (13)", # small_bowel
    14: "duodenum (14)",
    15: "colon (15)",
    16: "bladder (16)", # urinary_bladder
    17: "heart (17)",
    18: "left_lung (18)",
    19: "right_lung (19)"
}

MAP_WORD_TO_MATCHED_LABELS = {
    0: 0, # background
    1: 1, # liver
    2: 2, # spleen
    3: 3, # left kidney
    4: 4, # right kidney
    5: 5, # stomach
    6: 6, # gallbladder
    7: 7, # pancreas
    8: 8, # duodenum
    9: 9, # colon
    10: 10, # intestine
    11: 11, # left adrenal gland
    12: 0, # rectum (remove from dataset)
    13: 13, # bladder
    14: 14 # right adrenal gland
}

MAP_TOTAL_SEGMENTATOR_TO_MATCHED_LABELS = {
    0: 0, # background
    1: 2, # spleen
    2: 4, # right kidney
    3: 3, # left kidney
    4: 6, # gallbladder
    5: 1, # liver
    6: 5, # stomach
    7: 15, # aorta
    8: 7, # pancreas
    9: 14, # right adrenal gland
    10: 11, # left adrenal gland
    11: 16, # esophagus
    12: 12, # trachea 
    13: 10, # intestine
    14: 8, # duodenum
    15: 9, # colon
    16: 13, # bladder
    17: 17, # heart 
    18: 18, # left lung 
    19: 19 # right lung 
}

MAP_MATCHED_LABELS_ID_TO_LABEL_AND_ID = {
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
    12: "trachea (12)",
    13: "bladder (13)",
    14: "right adrenal gland (14)",
    15: "aorta (15)",
    16: "esophagus (16)",
    17: "heart (17)",
    18: "left lung (18)",
    19: "right lung (19)"
}

MAP_MATCHED_LABELS_ID_TO_LABEL = {
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
    12: "trachea",
    13: "bladder",
    14: "right adrenal gland",
    15: "aorta",
    16: "esophagus",
    17: "heart",
    18: "left lung",
    19: "right lung"
}


def check_bboxes(source_folder, dataset_name):
    if dataset_name == "WORD":
        dict_labels = WORD_MAP_ID_TO_LABEL
    elif dataset_name == "TOTAL_SEGMENTATOR":
        dict_labels = TOTAL_SEGMENTATOR_MAP_ID_TO_LABEL 
    elif dataset_name == "MATCHED_LABELS":
        dict_labels = MAP_MATCHED_LABELS_ID_TO_LABEL_AND_ID

    # Go through all the files in the source folder
    for root, dirs, files in os.walk(source_folder, topdown=True):
        files_npy = [f for f in files if f.endswith('.npy')] # Get all .npy files

        for file in files_npy: # For each .npy file
            if file == "label.npy":
                label = np.load(os.path.join(root, file))

                # Create bboxes for each label and check if the bboxes are correct   
                for label_id in dict_labels.keys():
                    if label_id == 0:
                        continue

                    label_bbox = np.squeeze(label == label_id).astype(np.uint8) 
                    if np.sum(label_bbox) == 0:
                        print(f"Label {label_id} in {os.path.join(root, file)} is not present.")
                        continue

                    # Compute region properties (including bounding box)
                    props = regionprops(label_bbox)[0]
                    bbox = props.bbox
                    assert bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] >= 0 and bbox[3] >= 0 and bbox[4] >= 0 and bbox[5] >= 0, f"bbox: {bbox} for label {label_id} in {os.path.join(root, file)} is not correct."
                    assert bbox[0] < bbox[3] and bbox[1] < bbox[4] and bbox[2] < bbox[5], f"bbox: {bbox} for label {label_id} in {os.path.join(root, file)} is not correct."
                    # print(f"Label {label_id} bbox: {bbox} in {os.path.join(root, file)} is correct.")

if __name__ == "__main__":
    
    source_folder = "./matched_datasets/"
    dataset_name = "MATCHED_LABELS"
    check_bboxes(source_folder, dataset_name)
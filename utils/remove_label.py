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

def remove_label(script_dir, label_value=None):
    if label_value is None:
        return None
    else:
        for dirpath, dirnames, filenames in os.walk(script_dir): 
            for filename in filenames:
                if filename == "label.npy":
                    label_path = os.path.join(dirpath, filename)
                    label = np.load(label_path)
                    label[label == label_value] = 0
                    np.save(label_path, label)
                    print(f"Label {label_value} removed and saved to {label_path}")
    print(f"Finished removing label {label_value}")


if __name__ == "__main__":
    np.random.seed(0)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Processing abdomen atlas data in {script_dir}")
    remove_label(script_dir, label_value=ORGAN_TO_LABEL["postcava"])
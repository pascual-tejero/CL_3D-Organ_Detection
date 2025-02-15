import os
import numpy as np

def check_sizes_abdomen_atalas(dataset_path):
    """
    Check the sizes of the images in the abdomen atlas dataset.
    """
    set_sizes_data = {"train": set(), "val": set(), "test": set()}
    set_sizes_label = {"train": set(), "val": set(), "test": set()}
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if filename == "data.npy":
                data = np.load(os.path.join(root, filename))
                if "train" in root:
                    set_sizes_data["train"].add(data.shape)
                elif "val" in root:
                    set_sizes_data["val"].add(data.shape)
                elif "test" in root:
                    set_sizes_data["test"].add(data.shape)
            elif filename == "label.npy":
                data = np.load(os.path.join(root, filename))
                if "train" in root:
                    set_sizes_label["train"].add(data.shape)
                elif "val" in root:
                    set_sizes_label["val"].add(data.shape)
                elif "test" in root:
                    set_sizes_label["test"].add(data.shape)

    print("Sizes of data images: ", set_sizes_data)
    print("Sizes of label images: ", set_sizes_label)

def check_labels_abdomen_atalas(dataset_path):
    """
    Check the labels of the images in the abdomen atlas dataset.
    """
    set_labels = {"train": set(), "val": set(), "test": set()}
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if filename == "label.npy":
                data = np.load(os.path.join(root, filename))
                if "train" in root:
                    set_labels["train"].update(np.unique(data))
                elif "val" in root:
                    set_labels["val"].update(np.unique(data))
                elif "test" in root:
                    set_labels["test"].update(np.unique(data))

    print("Labels in data images: ", set_labels)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    check_sizes_abdomen_atalas(script_dir)
    check_labels_abdomen_atalas(script_dir)
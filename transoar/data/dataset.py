"""Module containing the dataset related functionality."""

from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from transoar.data.transforms import get_transforms

class TransoarDataset(Dataset):
    """Dataset class of the transoar project."""
    def __init__(self, config, split, dataset=1, selected_samples=None, test_script=False):
        assert split in ['train', 'val', 'test']
        self._config = config
        self._split = split
        self._dataset = dataset
        self._selected_samples = selected_samples

        data_dir = Path(os.getenv("TRANSOAR_DATA")).resolve()

        if test_script: # Parameters for testing the model
            self._path_to_split = data_dir / self._config['dataset'] / split
            self._data = [data_path.name for data_path in self._path_to_split.iterdir()]

        elif config["mixing_datasets"] and split == "train": # Mix datasets and train on both
            self._path_to_split = data_dir / self._config['dataset'] / split
            self._path_to_split_2 = data_dir / self._config['dataset_2'] / split

            self._data = [] # Add samples alternatively from both datasets

            # Get all samples from the dataset folder
            list_dataset1 = [data_path.name for data_path in self._path_to_split.iterdir()]
            list_dataset2 = [data_path.name for data_path in self._path_to_split_2.iterdir()]

            if len(list_dataset1) <= len(list_dataset2):  # If dataset 1 has more samples than dataset 2
                self._path_to_split, self._path_to_split_2 = self._path_to_split_2, self._path_to_split

            # Determine which list is shorter and needs to be repeated
            short_dataset, long_dataset = (list_dataset1, list_dataset2) if len(list_dataset1) <= len(list_dataset2) else (list_dataset2, list_dataset1)

            # Repeat elements of the shorter list to match the length of the longer list
            repeated_short_list = short_dataset * (len(long_dataset) // len(short_dataset)) + short_dataset[:len(long_dataset) % len(short_dataset)]

            # Interleave the samples from both lists
            for idx in range(len(long_dataset)):
                self._data.append(long_dataset[idx])
                self._data.append(repeated_short_list[idx])


        else: # Rest of the cases
            dataset_key = 'dataset' if self._dataset == 1 else 'dataset_2'
            self._path_to_split = data_dir / self._config[dataset_key] / split
            self._data = [] # Add samples from the dataset

            if isinstance(self._selected_samples, dict): # CL_replay
                self._path_to_split = data_dir / self._config['dataset'] / split
                self._path_to_split_2 = data_dir / self._config['dataset_2'] / split

                # Get keys from selected samples dict in a list
                list_selected_samples = list(self._selected_samples.keys())

                # Few-shot training and the number of CL_replay samples is greater than the few-shot samples
                if config["few_shot_training"] and config["CL_replay_samples"] > config["few_shot_samples"]:
                    list_dataset1 = [data_path_dat.name for data_path_dat in self._path_to_split.iterdir()]
                    count = 0
                    for idx, data_path in enumerate(list_selected_samples):
                        self._data.append(list_dataset1[count])
                        self._data.append(data_path.parts[-1])
                        count += 1
                        if idx + 1 == config["CL_replay_samples"]:
                            break
                        if count == config["few_shot_samples"]:
                            count = 0

                # Rest of the cases (normal CL replay or few-shot training with more samples from the dataset)
                else:
                    count = 0
                    for idx, data_path in enumerate(self._path_to_split.iterdir()):
                        self._data.append(data_path.name)
                        self._data.append(list_selected_samples[count].parts[-1])
                        count += 1
                        if config["few_shot_training"] and idx + 1 == config["few_shot_samples"]: # Use only a few samples
                            break
                        if count == len(list_selected_samples):
                            count = 0   

            else:
                # Get all samples from the dataset folder
                self._data = [data_path.name for data_path in self._path_to_split.iterdir()]

                # Use only a few samples
                if config["few_shot_training"] and split == "train" and not config["CL_replay"]:
                    self._data = self._data[:config["few_shot_samples"]]


        self._augmentation = get_transforms(split, config, config["augmentation"]["apply_croping"])


    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self._config['overfit']:
            idx = 0

        case = self._data[idx] # Get the case name

        # Mix datasets and train on both
        if self._config["mixing_datasets"] and self._split == "train" or isinstance(self._selected_samples, dict):
            if idx % 2 == 0: 
                path_to_case = self._path_to_split / case # Dataset task 1
            else:
                path_to_case = self._path_to_split_2 / case # Dataset task 2

        else:
            path_to_case = self._path_to_split / case # Normal training

        data_path, label_path = sorted(list(path_to_case.iterdir()), key=lambda x: len(str(x))) # Sort the files by length of the string

        # Load npy files
        data, label = np.load(data_path), np.load(label_path)

        # Apply data augmentation
        if self._config['augmentation']['use_augmentation']:
            data_dict = {
                'image': data,
                'label': label
            }

            # Apply data augmentation
            self._augmentation.set_random_state(torch.initial_seed() + idx)

            data_transformed = self._augmentation(data_dict)
            data, label = data_transformed['image'], data_transformed['label']
        else:
            data, label = torch.tensor(data), torch.tensor(label)
        # print("data, label", data.shape, label.shape)
        
        
        if self._split == 'test': # Return path to case for visualization
            return data, label, path_to_case # path is used for visualization of predictions on source data
        elif self._config["CL_replay"] and self._split == "train" and self._dataset == 2 and self._selected_samples is None:
            return data, label, path_to_case
        else:
            return data, label # Return data and label

"""Module containing the dataset related functionality."""

from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from transoar.data.transforms import get_transforms

#data_base_dir = "/mnt/data/transoar_prep/dataset/"  #"datasets/"

class TransoarDataset(Dataset):
    """Dataset class of the transoar project."""
    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        self._config = config
        data_dir = Path(os.getenv("TRANSOAR_DATA")).resolve()
        self._path_to_split = data_dir / self._config['dataset'] / split
        self._split = split
        self._data = []
        #if split == 'train':
        #    read_limit = 120
        #elif split == 'val':
        #    read_limit = 20
        #elif split == 'test':
        #    read_limit = 20
        #for data_path in self._path_to_split.iterdir():
        #    if len(self._data) < read_limit:
        #        self._data.append(data_path.name)
        #print("limited data read for ", split, ": ", len(self._data))
        
        self._data = [data_path.name for data_path in self._path_to_split.iterdir()]

        self._augmentation = get_transforms(split, config)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self._config['overfit']:
            idx = 0

        case = self._data[idx]
        path_to_case = self._path_to_split / case
        data_path, label_path = sorted(list(path_to_case.iterdir()), key=lambda x: len(str(x)))

        # Load npy files
        data, label = np.load(data_path), np.load(label_path)

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

        if self._split == 'test':
            return data, label, path_to_case # path is used for visualization of predictions on source data
        else:
            return data, label

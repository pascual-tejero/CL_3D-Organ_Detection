"""Module containing the dataset related functionality."""

from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from transoar.data.patch_transforms import get_transforms
from transoar.utils.bboxes import segmentation2bbox
import monai
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
        successful_crop = False
        case = self._data[idx]
        path_to_case = self._path_to_split / case
        data_path, label_path = sorted(list(path_to_case.iterdir()), key=lambda x: len(str(x)))

        # Load npy files
        data, label = np.load(data_path), np.load(label_path)
        #print("pre-aug", data.shape)

        if self._config['augmentation']['use_augmentation']:
            data_dict = {
                'image': data,
                'label': label
            }
            # Apply data augmentation
            self._augmentation.set_random_state(torch.randint(0,2**30,(1,)).item())
            if self._split == 'train': # added check for non-empty patches for training
                padding_size = self.gen_padding_size(label, self._config['augmentation']['patch_size'][0],
                                                     self._config['augmentation']['stride'])
                scale_int = monai.transforms.ScaleIntensityRanged(
                            # keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
                            keys=['image'], a_min=self._config['foreground_voxel_statistics']['percentile_00_5'], 
                            a_max=self._config['foreground_voxel_statistics']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True
                            )
                pad = monai.transforms.SpatialPadd( # patches also contain padded space like in val or test
                        keys=['image', 'label'],
                        spatial_size=padding_size # this is indiv. for every image, → can't be in the regular augmentator
                    )
                data_transformed = scale_int(data_dict)
                data_transformed = pad(data_transformed)
                try_counter = 0
                while(not successful_crop and try_counter < 5): # try 5 random crops, if no organs found, last random crop w/o organs used
                    data_transformed = self._augmentation(data_transformed) # random crop + regular augmentation
                    data, label = data_transformed['image'], data_transformed['label']
                    batch_bboxes, batch_classes = segmentation2bbox(label[None,:], self._config['bbox_padding'], excl_crossed_boundary=True)
                    try_counter += 1
                    for bc in batch_classes:
                        if bc.numel():  # if tensor not empty
                            successful_crop = True

            else:
                data_transformed = self._augmentation(data_dict)
                data, label = data_transformed['image'], data_transformed['label']
                            

        else:
            data, label = torch.tensor(data), torch.tensor(label)
        #print("post-aug", data.shape)
        if self._split == 'test':
            return data, label, path_to_case # path is used for visualization of predictions on source data
        else:
            return data, label
        
    def gen_padding_size(self, input_tensor, patch_size, stride):
        padding = []
        input_tensor = torch.tensor(input_tensor)
        for idx, s in enumerate(input_tensor.shape[-3:]):
            pd = s%stride
            pd = 0 if pd == 0 else stride-pd
            if (s+pd) < patch_size:  # check if one patch fits
                pd = patch_size - s  # add padding to get min. patch size
            padding.append(pd)
        padding.reverse()

        padding_both_sides = []
        for size in padding:
            padding_both_sides.extend([size // 2, size - size // 2])
        padded_tensor = torch.nn.functional.pad(input_tensor, padding_both_sides, mode='constant', value=0)
        padded_size = padded_tensor.shape[-3:]
        return padded_size

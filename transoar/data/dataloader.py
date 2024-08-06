"""Module containing dataloader related functionality."""

import torch
from torch.utils.data import DataLoader

from transoar.data.dataset import TransoarDataset
from transoar.utils.bboxes import segmentation2bbox

def get_loader(config, split, batch_size=None, test_script=False):
    batch_size = batch_size or config['batch_size'] # Default batch size
    shuffle = False if split in ['test', 'val'] else config['shuffle'] # Shuffle only for training 
    collator = TransoarCollator(config, split, CL_replay=test_script or (split == 'train' and config.get("CL_replay", False)))

    # Test script
    if test_script:
        dataset = TransoarDataset(config, split, dataset=1, selected_samples=None, test_script=True)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config['num_workers'], collate_fn=collator)

    # Test split if CL_reg, CL_replay or mixing_datasets is True
    if split == 'test' and (config.get("CL_reg") or config.get("CL_replay") or config.get("mixing_datasets") or config.get("test")):
        dataset_1 = TransoarDataset(config, split, dataset=1)
        dataset_2 = TransoarDataset(config, split, dataset=2)
        dataloader_1 = DataLoader(dataset_1, batch_size=batch_size, shuffle=shuffle, num_workers=config['num_workers'], collate_fn=collator)
        dataloader_2 = DataLoader(dataset_2, batch_size=batch_size, shuffle=shuffle, num_workers=config['num_workers'], collate_fn=collator)
        return (dataloader_1, dataloader_2)

    # CL_reg training or validation
    if config.get("CL_reg") and not config.get("CL_replay") and not config.get("mixing_datasets"):
        dataset = TransoarDataset(config, split, dataset=1) # Take the dataset of task 2 for CL_reg
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config['num_workers'], collate_fn=collator)

    # CL_replay training or validation
    if config.get("CL_replay") and not config.get("CL_reg") and not config.get("mixing_datasets"):
        dataset = TransoarDataset(config, split, dataset=2 if split == 'train' else 1) # Take the dataset of task 1 for CL_replay
        batch_size = 1 if split == 'train' else batch_size
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config['num_workers'], collate_fn=collator)

    # Mixing datasets training
    if config.get("mixing_datasets") and not config.get("CL_reg") and not config.get("CL_replay"):
        dataset = TransoarDataset(config, split) # Normal training
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=config['num_workers'], collate_fn=collator)

    dataset = TransoarDataset(config, split) # Normal training

    # Return dataloader with collator
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config['num_workers'], collate_fn=collator)



def get_loader_CLreplay_selected_samples(config, split, batch_size=None, selected_samples=None):

    # Init collator
    collator = TransoarCollator(config, split)
    shuffle = False 

    dataset = TransoarDataset(config, split, dataset=1, selected_samples=selected_samples)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=config['num_workers'], collate_fn=collator
    )

    return dataloader

# def init_fn(worker_id):
#     """
#     https://github.com/pytorch/pytorch/issues/7068
#     https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
#     """
#     torch_seed = torch.initial_seed()
#     if torch_seed >= 2**30:
#         torch_seed = torch_seed % 2**30
#     seed = torch_seed + worker_id

#     random.seed(seed)   
#     np.random.seed(seed)
#     monai.utils.set_determinism(seed=seed)
#     torch.manual_seed(seed)


class TransoarCollator:
    def __init__(self, config, split, CL_replay=False):
        self._bbox_padding = config['bbox_padding']
        self._split = split
        self.CL_replay = CL_replay

    def __call__(self, batch):
        batch_images = []
        batch_labels = []
        batch_masks = []
        if self._split == 'test' or self.CL_replay:
            batch_paths = []
            for image, label, path in batch:
                batch_images.append(image)
                batch_labels.append(label)
                batch_masks.append(torch.zeros_like(image))
                batch_paths.append(path)
        else:
            for image, label in batch:
                batch_images.append(image)
                batch_labels.append(label)
                batch_masks.append(torch.zeros_like(image))

        # Generate bboxes and corresponding class labels
        batch_bboxes, batch_classes = segmentation2bbox(torch.stack(batch_labels), self._bbox_padding)
        # print("batch_bboxes, batch_classes", batch_bboxes, batch_classes)
        # quit()

        if self._split == 'test' or self.CL_replay:
            return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels), batch_paths    
        return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels)

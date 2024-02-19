"""Module containing dataloader related functionality."""

import torch
from torch.utils.data import DataLoader

from transoar.data.dataset import TransoarDataset
from transoar.utils.bboxes import segmentation2bbox


def get_loader(config, split, batch_size=None):
    if not batch_size:
        batch_size = config['batch_size']

    # Init collator
    collator = TransoarCollator(config, split)
    shuffle = False if split in ['test', 'val'] else config['shuffle']

    dataset = TransoarDataset(config, split)
    # for i in range(1000):   
    #     dataset.__getitem__(i)

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
    def __init__(self, config, split):
        self._bbox_padding = config['bbox_padding']
        self._split = split

    def __call__(self, batch):
        batch_images = []
        batch_labels = []
        batch_masks = []
        if self._split == 'test':
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

        if self._split == 'test':
            return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels), batch_paths    
        return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels)

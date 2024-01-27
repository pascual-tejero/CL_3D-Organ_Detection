"""Module containing dataloader related functionality."""

import torch
from torch.utils.data import DataLoader

from transoar.data.patch_dataset import TransoarDataset
from transoar.utils.bboxes import segmentation2bbox
try:
    import matplotlib.pyplot as plt
except:
    print("matplotlib could not be imported")


def get_loader(config, split, batch_size=None):
    if not batch_size:
        batch_size = config['batch_size']

    # Init collator
    collator = TransoarCollator(config, split)
    shuffle = False if split in ['test', 'val'] else config['shuffle']

    dataset = TransoarDataset(config, split)
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
        self._stride = config['augmentation']['stride']
        patch_size = config['augmentation']['patch_size']
        self._patch_size = patch_size[0]
        assert all(element == self._patch_size for element in patch_size), "Only patches with same size along each axis supported right now!"

    def __call__(self, batch):
        batch_images = []
        batch_labels = []
        batch_masks = []
        if self._split == 'test':
            assert len(batch) == 1
            batch_paths = []
            for image, label, path in batch:
                # Trick: since batch size is 1, we simple use the batch dimension to stack patches
                batch_images, padded_batch_images, patch_offsets = self.gen_patches(image, self._stride, self._patch_size)
                test_patch_mask, batch_labels, _ = self.gen_patches(label, self._stride, self._patch_size)
                batch_masks.append(test_patch_mask)
                #batch_images, batch_labels = self.filter_empty_patches(batch_images, batch_labels)
                                            
                batch_images = batch_images.chunk(batch_images.shape[0], dim=0)
                batch_labels = batch_labels.chunk(batch_labels.shape[0], dim=0)
                batch_paths.append(path)

        elif self._split == 'val':
            assert len(batch) == 1
            for image, label in batch:
                # Trick: since batch size is 1, we simple use the batch dimension to stack patches
                batch_images, _, _ = self.gen_patches(image, self._stride, self._patch_size)
                batch_labels, padded_batch_labels, patch_offsets = self.gen_patches(label, self._stride, self._patch_size)
                batch_masks.append(torch.zeros_like(image))
                #batch_images, batch_labels, patch_offsets = self.filter_empty_patches(batch_images, batch_labels, patch_offsets)
                                            
                batch_images = batch_images.chunk(batch_images.shape[0], dim=0)
                batch_labels = batch_labels.chunk(batch_labels.shape[0], dim=0)
                # vizualization for patches
                #plt.imshow(batch_images[0][0,:,:,96], cmap='gray')
                #plt.savefig('my_slice.png')
                #plt.imshow(batch_labels[0][0,:,:,96], cmap='Spectral')
                #plt.savefig('my_slice2.png')
        else:
            for image, label in batch:
                batch_images.append(image)
                batch_labels.append(label)
                batch_masks.append(torch.zeros_like(image))
                
                #plt.imshow(label[0,:,:,96], cmap='Spectral')
                #plt.savefig(f'z_slice.png')
                

        
        if self._split == 'val':
            batch_bboxes, batch_classes = [],[]
            batch_bboxes_tmp, batch_classes_tmp = segmentation2bbox(torch.stack(batch_labels), self._bbox_padding, excl_crossed_boundary=True)
            whole_padded_boxes, whole_padded_classes = segmentation2bbox(torch.stack([padded_batch_labels]), self._bbox_padding)
            batch_labels_tmp, batch_images_tmp, patch_offsets_tmp = [],[],[]
            b_classes = []
            for idx, bc in enumerate(batch_classes_tmp): # iterate patches
                #if bc.numel():  # if tensor not empty
                    # only keep non-empty patches
                batch_bboxes.append(batch_bboxes_tmp[idx])
                batch_classes.append(bc)
                batch_labels_tmp.append(batch_labels[idx])
                batch_images_tmp.append(batch_images[idx])
                patch_offsets_tmp.append(patch_offsets[idx])
                b_classes.extend(torch.unique(bc).tolist())  # add class ids present in patch
            batch_labels, batch_images, patch_offsets = tuple(batch_labels_tmp), tuple(batch_images_tmp), torch.stack(patch_offsets_tmp)
            b_classes = set(b_classes)  # get ids which are present in the current image
            """with open(f'cls_val_sliding_win_{self._patch_size}_{self._stride}.txt', 'a') as fp:
                for item in b_classes:
                    fp.write("%s\n" % item)"""
        else:
            # Generate bboxes and corresponding class labels
            batch_bboxes, batch_classes = segmentation2bbox(torch.stack(batch_labels), self._bbox_padding)
            """with open(f'cls_rdm_crop_training_{self._patch_size}.txt', 'a') as fp:
                for b_classes in batch_classes:
                    for item in b_classes:
                        fp.write("%s\n" % item.item())"""
        if self._split == 'test':
            #batch_bboxes, batch_classes = segmentation2bbox(torch.stack(batch_labels), self._bbox_padding)
            batch_mask_bboxes, batch_mask_classes = segmentation2bbox(torch.stack(batch_masks).permute(1,0,2,3,4), self._bbox_padding)
            return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels), batch_paths, patch_offsets, padded_batch_images, list(zip(batch_mask_bboxes, batch_mask_classes))
        elif self._split == 'val':
            # returns patch images, _, patch boxes, patch labels, patch offsets, whole padded labels, whole_padded_boxes, whole_padded_classes
            return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels), patch_offsets, padded_batch_labels, list(zip(whole_padded_boxes, whole_padded_classes))
        return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels)

    def gen_patches(self, input_tensor, stride, patch_size):
        # Calculate the padding needed for each dimension
        assert len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1  # (bs, ch, h, w, d) and bs == 1
        padding = []
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
        def extract_patches(input_tensor, patch_size, stride_size):
            patches = input_tensor.unfold(1, patch_size, stride_size)
            patches = patches.unfold(2, patch_size, stride_size)
            patches = patches.unfold(3, patch_size, stride_size)
            patches = patches.contiguous()
            patches = patches.view(-1, patch_size, patch_size, patch_size)
            return patches
        patches = extract_patches(padded_tensor, patch_size, stride)
        grid = torch.meshgrid([torch.arange(0, size - patch_size + 1, stride) for size, patch_size in zip(padded_tensor.shape[-3:], [patch_size,patch_size,patch_size])])
        offsets = torch.stack(grid, dim=-1).reshape(-1, 3)
        assert patches.shape[0] == offsets.shape[0], "error in offset computation"
        return patches, padded_tensor, offsets

    def filter_empty_patches(self, batch_images, batch_labels, patch_offsets):
        mask = torch.sum(batch_labels, dim=(1,2,3)) != 0
        # Use this mask to select the non-zero channels from the tensor
        batch_images = batch_images[mask]
        batch_labels = batch_labels[mask]
        patch_offsets = patch_offsets[mask]
        assert batch_images.shape[0] and batch_labels.shape[0] and patch_offsets.shape[0]
        return batch_images, batch_labels, patch_offsets

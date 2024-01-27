"""Script to visualize predictions on slices."""

import os,sys
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="TypedStorage")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("append to path & chdir:", base_dir)
os.chdir(base_dir)
sys.path.append(base_dir)


from transoar.utils.io import load_json, write_json
from transoar.utils.visualization import save_attn_visualization, save_pred_visualization
from transoar.data.dataloader import get_loader
from transoar.models.transoarnet import TransoarNet
from transoar.evaluator import DetectionEvaluator
from transoar.inference import inference
from transoar.utils.bboxes import segmentation2bbox, box_cxcyczwhd_to_xyzxyz

# plt visualization
import os
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.widgets import Slider
from matplotlib.widgets import CheckButtons

import scipy.ndimage

class Tester_w_plt:

    def __init__(self, args):
        path_to_run = Path('./runs/' + args.run)
        config = load_json(path_to_run / 'config.json')

        self._save_preds = args.save_preds
        self._save_attn_map = args.save_attn_map
        self._full_labeled = args.full_labeled
        self._headless = args.headless
        self._class_dict = config['labels']

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_gpu)
        self._device = 'cuda' if args.num_gpu >= 0 else 'cpu'

        # Get path to checkpoint
        avail_checkpoints = [path for path in path_to_run.iterdir() if 'model_' in str(path)]
        avail_checkpoints.sort(key=lambda x: len(str(x)))
        if args.last:
            path_to_ckpt = avail_checkpoints[0]
        else:
            path_to_ckpt = avail_checkpoints[-1]

        # Build necessary components
        self._set_to_eval = 'val' if args.val else 'test'
        self._test_loader = get_loader(config, self._set_to_eval, batch_size=1)

        self._model = TransoarNet(config).to(device=self._device)

        # Load checkpoint
        checkpoint = torch.load(path_to_ckpt, map_location=self._device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()

        # Create dir to store results
        self._path_to_results = path_to_run / 'results' / path_to_ckpt.parts[-1][:-3]
        self._path_to_results.mkdir(parents=True, exist_ok=True)

        if self._save_preds:
            self._path_to_vis = self._path_to_results / ('vis_' + self._set_to_eval)
            self._path_to_vis.mkdir(parents=False, exist_ok=True)
  
    def run(self):
        if self._save_attn_map:
            backbone_features_list, dec_attn_weights_list = [], []
            
            # Register hooks to efficiently access relevant weights
            hooks = [
                self._model._backbone.P2_conv2.register_forward_hook(
                    lambda self, input, output: backbone_features_list.append(output)
                ),
                self._model._neck.decoder.layers[-1].cross_attn.register_forward_hook(
                    lambda self, input, output: dec_attn_weights_list.append(output[1])
                )
            ]
    
        with torch.no_grad():
            for idx, (data, mask, bboxes, seg_mask, paths) in enumerate(tqdm(self._test_loader)):
                # Put data to gpu
                data, mask = data.to(device=self._device), mask.to(device=self._device)
            
                targets = {
                    'boxes': bboxes[0][0].to(dtype=torch.float, device=self._device),
                    'labels': bboxes[0][1].to(device=self._device)
                }

                # Only use complete data for performance evaluation
                #if targets['labels'].shape[0] < len(self._class_dict):
                #    continue

                # Make prediction
                out = self._model(data)

                # Format out to fit evaluator and estimate best predictions per class
                pred_boxes, pred_classes, pred_scores = inference(out)
                gt_boxes = [targets['boxes'].detach().cpu().numpy()]
                gt_classes = [targets['labels'].detach().cpu().numpy()]
               
                for i in range(len(pred_boxes)):
                    #xyzxyz_scaling = np.array([data[i].shape[-3:],data[i].shape[-3:]]).flatten() # for undoing normalization
                    xyzxyz_scaling = [128,128,128,128,128,128] # fixed resolution for visualization
                    pd_b = box_cxcyczwhd_to_xyzxyz(pred_boxes[i]*xyzxyz_scaling)
                    gt_b = box_cxcyczwhd_to_xyzxyz(gt_boxes[i]*xyzxyz_scaling)
                    vis_axial_slide(pd_b, gt_b, paths[i], self._headless)

                inp = input("next image? (y or n): ")
                if inp == "n" or inp == "no":
                    sys.exit()


def vis_axial_slide(box1_list,box2_list,path, headless=True):
    data = np.load(os.path.join(path,"data.npy"))[0]

    x, y, z = data.shape
    new_shape = (128, 128, 128)

    zoom_factors = (new_shape[0]/x, new_shape[1]/y, new_shape[2]/z)
    data = scipy.ndimage.zoom(data, zoom_factors, order=3)

    # Create a 1x2 grid of subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax4.axis('off')
    # toggle
    labels = ['Index ' + str(i+1) for i, n in enumerate(box1_list)]
    states = [True] * len(box1_list)
    checkboxes = CheckButtons(ax4, labels,states)
    def toggle_indices(label):
        index = labels.index(label)
        states[index] = not states[index]
        update(0)

    checkboxes.on_clicked(toggle_indices)


    # Plot the initial axial center slice on the first subplot
    im = ax1.imshow(data[:, :, data.shape[2] // 2].T, cmap="gray", origin="lower")
    im_x = ax2.imshow(data[data.shape[0] // 2, :, :].T, cmap="gray", origin="lower")
    im_y = ax3.imshow(data[:, data.shape[1] // 2, :].T, cmap="gray", origin="lower")


    if not headless:
        # Add slider axis
        slider_ax = fig.add_axes([0.2, 0.11, 0.6, 0.03], facecolor='yellow')
        slider = Slider(slider_ax, 'Z Slice', 0, data.shape[2]-1, valinit=data.shape[2] // 2, valstep=1)
        slider_ax_x = fig.add_axes([0.2, 0.01, 0.6, 0.03], facecolor='yellow')
        slider_x = Slider(slider_ax_x, 'X Slice', 0, data.shape[0]-1, valinit=data.shape[0] // 2, valstep=1)
        slider_ax_y = fig.add_axes([0.2, 0.06, 0.6, 0.03], facecolor='yellow')
        slider_y = Slider(slider_ax_y, 'Y Slice', 0, data.shape[1]-1, valinit=data.shape[1] // 2, valstep=1)
        plt.subplots_adjust(bottom=0.2)


    def update(val):
        # Update axial center slice based on slider position
        if headless:
            slice_idx = int(val)
            slice_idx_x = int(val)
            slice_idx_y = int(val)
        else:
            slice_idx = int(slider.val)
            slice_idx_x = int(slider_x.val)
            slice_idx_y = int(slider_y.val)

        im.set_data(data[:, :, slice_idx].T)
        im_x.set_data(data[slice_idx_x, :, :].T)
        im_y.set_data(data[:, slice_idx_y, :].T)



        # Clear existing patches (boxes)
        for patch in ax1.patches:
            patch.remove()
        for patch in ax2.patches:
            patch.remove()
        for patch in ax3.patches:
            patch.remove()

        # Redraw boxes in box1_list
        for box1 in box1_list[states]:
            if box1[2] <= slice_idx <= box1[5]:   # for z slice
                width1 = box1[3] - box1[0]
                height1 = box1[4] - box1[1]
                rect1_z = patches.Rectangle((box1[0], box1[1]), width1, height1, linewidth=1, edgecolor='r', facecolor='none', label='Pred')
                ax1.add_patch(rect1_z)

            if box1[0] <= slice_idx_x <= box1[3]:   # for x slice
                depth1 = box1[5] - box1[2]
                height1 = box1[4] - box1[1]
                rect1_x = patches.Rectangle((box1[2], box1[1]), depth1, height1, linewidth=1, edgecolor='r', facecolor='none', label='Pred')
                ax2.add_patch(rect1_x)

            if box1[1] <= slice_idx_y <= box1[4]:   # for y slice
                width1 = box1[3] - box1[0]
                depth1 = box1[5] - box1[2]
                rect1_y = patches.Rectangle((box1[0], box1[2]), width1, depth1, linewidth=1, edgecolor='r', facecolor='none', label='Pred')
                ax3.add_patch(rect1_y)


        # Redraw boxes in box2_list
        for box2 in box2_list[states]:
            if box2[2] <= slice_idx <= box2[5]:   # for z slice
                width1 = box2[3] - box2[0]
                height1 = box2[4] - box2[1]
                rect2_z = patches.Rectangle((box2[0], box2[1]), width1, height1, linewidth=1, edgecolor='g', facecolor='none', label='GT')
                ax1.add_patch(rect2_z)

            if box2[0] <= slice_idx_x <= box2[3]:   # for x slice
                depth1 = box2[5] - box2[2]
                height1 = box2[4] - box2[1]
                rect2_x = patches.Rectangle((box2[2], box2[1]), depth1, height1, linewidth=1, edgecolor='g', facecolor='none', label='GT')
                ax2.add_patch(rect2_x)

            if box2[1] <= slice_idx_y <= box2[4]:   # for y slice
                width1 = box2[3] - box2[0]
                depth1 = box2[5] - box2[2]
                rect2_y = patches.Rectangle((box2[0], box2[2]), width1, depth1, linewidth=1, edgecolor='g', facecolor='none', label='GT')
                ax3.add_patch(rect2_y)

        try:
            ax4.legend(handles=[rect1_z, rect2_z], loc='center')
        except:
            pass

        fig.canvas.draw_idle()

    if headless:
        nr=3
        random_idx = np.random.randint(1,data.shape[2]-1,nr)
        for i in random_idx:
            update(i)
            plt.savefig("img"+str(i)+".png")
    else:
        slider.on_changed(update)
        slider_x.on_changed(update)
        slider_y.on_changed(update)

        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    matplotlib.use('TkAgg')
    # Add necessary args
    parser.add_argument('--run', required=True, type=str, help='Name of experiment in transoar/runs.')
    parser.add_argument('--num_gpu', type=int, default=-1, help='Use model_last instead of model_best.')
    parser.add_argument('--val', action='store_true', help='Evaluate performance on test set.')
    parser.add_argument('--last', action='store_true', help='Use model_last instead of model_best.')
    parser.add_argument('--headless', action='store_true', help='Use model_last instead of model_best.')
    parser.add_argument('--save_preds', action='store_true', help='Save predictions.')
    parser.add_argument('--save_attn_map', action='store_true', help='Saves attention maps.')
    parser.add_argument('--full_labeled', action='store_true', help='Use only fully labeled data.')
    parser.add_argument('--coco_map', action='store_true', help='Use coco map.')
    args = parser.parse_args()

    tester = Tester_w_plt(args)
    tester.run()

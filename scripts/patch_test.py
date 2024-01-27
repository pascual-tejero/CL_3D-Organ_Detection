"""Script to evalute performance on the val and test set."""

import os,sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="TypedStorage")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("append to path & chdir:", base_dir)
os.chdir(base_dir)
sys.path.append(base_dir)


from transoar.utils.io import load_json, write_json
try:
    from transoar.utils.visualization import save_attn_visualization, save_pred_visualization
except:
    pass
from transoar.data.patch_dataloader import get_loader
from transoar.models.transoarnet import TransoarNet
from transoar.evaluator import DetectionEvaluator, SegmentationEvaluator
from transoar.inference import inference
from transoar.utils.bboxes import merge_patches
from transoar.utils.bboxes import box_cxcyczwhd_to_xyzxyz, iou_3d
from scripts.train import match

class Tester:

    def __init__(self, args):
        path_to_run = Path('./runs/' + args.run)
        self.config = load_json(path_to_run / 'config.json')

        self._save_preds = args.save_preds
        #self._save_attn_map = args.save_attn_map
        self._per_sample_results = args.per_sample_results
        self._per_patch = args.per_patch
        self._class_dict = self.config['labels']
        self._segm_eval = False #self.config['backbone']['use_seg_proxy_loss']

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
        self._test_loader = get_loader(self.config, self._set_to_eval, batch_size=1)

        self._evaluator = DetectionEvaluator(
            classes=list(self.config['labels'].values()),
            classes_small=self.config['labels_small'],
            classes_mid=self.config['labels_mid'],
            classes_large=self.config['labels_large'],
            iou_range_nndet=(0.1, 0.5, 0.05),
            iou_range_coco=(0.5, 0.95, 0.05),
            sparse_results=False
        )

        #self._segm_evaluator = SegmentationEvaluator(seg_fg_bg=self.config['backbone']['fg_bg'],
        #                                             ce_dice=self._segm_eval, 
        #                                             hd95=self._segm_eval)

        self._model = TransoarNet(self.config).to(device=self._device)

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
        per_sample_results = {}
        with torch.no_grad():
            for idx, (data, mask, bboxes, seg_mask, paths, patch_pos, padded_img, patch_bbox) in enumerate(tqdm(self._test_loader)):
                
                assert seg_mask[0].shape == padded_img.shape, f"{seg_mask.shape} - {padded_img.shape}"
                # Put data to gpu
                inf_out_patches = {} # track predictions for all patches of one image
               
                for ch in range(data.shape[0]): # iterate
                    patch_targets = {
                        'boxes': patch_bbox[ch][0].to(dtype=torch.float, device=self._device),
                        'labels': patch_bbox[ch][1].to(device=self._device)
                    }
                    patch_gt_boxes = [patch_targets['boxes'].detach().cpu().numpy()]
                    patch_gt_classes = [patch_targets['labels'].detach().cpu().numpy()]

                    data_c = data[ch,:].to(device=self._device)[None,:]
                    
                    # Make prediction
                    out = self._model(data_c)

                    # Format out to fit evaluator and estimate best predictions per class
                    pred_boxes, pred_classes, pred_scores = inference(out)
                    inf_out_patches[ch] = {"pred_boxes": pred_boxes[0],
                                       "pred_classes": pred_classes[0],
                                       "pred_scores": pred_scores[0]}
                    if self._save_preds and self._per_patch:
                        save_pred_visualization(
                            pred_boxes[0], pred_classes[0], patch_gt_boxes[0], patch_gt_classes[0], mask[0][ch][None,:], 
                            self._path_to_vis, self._class_dict, str(idx)+"-"+str(ch)
                        )
                    if self._per_sample_results and self._per_patch:
                        sample_name = paths[0].stem + f'_case{idx}-{ch}'
                        per_sample_results[sample_name] = self.export_per_sample_results(sample_name,
                                                                                         pred_classes,
                                                                                         patch_gt_classes,
                                                                                         pred_boxes,
                                                                                         patch_gt_boxes)
                # Merge patches
                #print("patch pos: ",patch_pos)
                #print("img size: ",padded_img.shape[-3:])
                pred_boxes, pred_classes, pred_scores = merge_patches(inf_out_patches, patch_pos, self.config['augmentation']['patch_size'], padded_img.shape[-3:], mode=self.config.get('patch_merge_mode', 'custom'), config=self.config)
                targets = {
                        'boxes': bboxes[0][0].to(dtype=torch.float, device=self._device),
                        'labels': bboxes[0][1].to(device=self._device)
                    }
                gt_boxes = [targets['boxes'].detach().cpu().numpy()]
                gt_classes = [targets['labels'].detach().cpu().numpy()]

                # Add pred to evaluator
                self._evaluator.add(
                    pred_boxes=pred_boxes,
                    pred_classes=pred_classes,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes
                )

                if self._per_sample_results:
                    sample_name = paths[0].stem + f'_case{idx}'
                    per_sample_results[sample_name] = self.export_per_sample_results(sample_name,
                                                                                     pred_classes,
                                                                                     gt_classes,
                                                                                     pred_boxes,
                                                                                     gt_boxes)
                    

                if self._save_preds:
                    save_pred_visualization(
                        pred_boxes[0], pred_classes[0], gt_boxes[0], gt_classes[0], seg_mask[0], 
                        self._path_to_vis, self._class_dict, idx
                    )

                #break

            # Get and store final results
            # [torch.tensor([id_ for score, id_ in query_info[c]]).unique().shape for c in query_info.keys()]
            metric_scores = self._evaluator.eval()
            if self._segm_eval:
                metrics_segmentation = self._segm_evaluator.eval(out,seg_mask)
                metric_scores.update(metrics_segmentation)

            # Analysis of model parameter distribution
            num_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            num_backbone_params = sum(p.numel() for n, p in self._model.named_parameters() if p.requires_grad and match(n, ['backbone', 'input_proj', 'skip']))
            num_neck_params = sum(p.numel() for n, p in self._model.named_parameters() if p.requires_grad and match(n, ['neck', 'query']))
            num_head_params = sum(p.numel() for n, p in self._model.named_parameters() if p.requires_grad and match(n, ['head']))

            num_params_dict ={'num_params': num_params,
                      'num_backbone_params': num_backbone_params,
                      'num_neck_params': num_neck_params,
                      'num_head_params': num_head_params
                      }
            metric_scores.update(num_params_dict)  # Add parameters to result log

            write_json(metric_scores, self._path_to_results / ('results_' + self._set_to_eval + '.json'))
            if self._per_sample_results:
                write_json(per_sample_results, self._path_to_results / ('per_sample_results_' + self._set_to_eval + '.json'))

    def export_per_sample_results(self, sample_name, pred_classes, gt_classes, pred_boxes, gt_boxes):
        assert len(gt_classes) == 1 # batch size == 1
        keys_as_int = [int(key) for key in self.config['labels'].keys()]
        unique_classes = np.unique(keys_as_int)
        tmp_dict = {}
        for class_ in unique_classes:
            pred_miss = False
            gt_miss = False
            pred_id = np.where(pred_classes[0]==class_)
            if len(pred_id[0]) == 0: # no prediction
                pred_miss = True
            else:
                pred_id = pred_id[0]
            gt_id = np.where(gt_classes[0]==class_)
            if len(gt_id[0]) == 0: # no gt
                gt_miss = True
            else:
                gt_id = gt_id[0]
            if gt_miss and pred_miss:
                result = 'TN'
            elif gt_miss:
                result = 'FP'
                if not self._per_patch: print(f"no GT for {class_} in {sample_name}")
            elif pred_miss:
                result = 'FN'
                if not self._per_patch: print(f"no prediction for {class_} in {sample_name}")
            else: # return IOU for TP
                pred_box = torch.tensor(pred_boxes[0][pred_id])
                gt_box = torch.tensor(gt_boxes[0][gt_id])
                result,_ = iou_3d(box_cxcyczwhd_to_xyzxyz(pred_box), box_cxcyczwhd_to_xyzxyz(gt_box))
                result = str(result.item())[:4] + ' IoU'
            tmp_dict[self.config['labels'][str(class_)]] = result
        return tmp_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add necessary args
    parser.add_argument('--run', required=True, type=str, help='Name of experiment in transoar/runs.')
    parser.add_argument('--num_gpu', type=int, default=-1, help='Use model_last instead of model_best.')
    parser.add_argument('--val', action='store_true', help='Evaluate performance on test set.')
    parser.add_argument('--last', action='store_true', help='Use model_last instead of model_best.')
    parser.add_argument('--save_preds', action='store_true', help='Save predictions.')
    #parser.add_argument('--save_attn_map', action='store_true', help='Saves attention maps.') # not implemented for patch-based
    parser.add_argument('--per_sample_results', action='store_true', help='Saves per sample results of predictions.')
    parser.add_argument('--per_patch', action='store_true', help='If per_sample_results is set evals results for each patch.\
                                                                  If save_preds is set → generates visualizations for every patch.')    
    
    args = parser.parse_args()

    tester = Tester(args)
    tester.run()

"""Module containing the trainer of the transoar project."""

import copy
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transoar.evaluator import DetectionEvaluator
from transoar.inference import inference
from transoar.utils.bboxes import merge_patches

class Trainer:

    def __init__(
        self, train_loader, val_loader, model, criterion, optimizer, scheduler,
        device, config, path_to_run, epoch, metric_start_val
    ):
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._path_to_run = path_to_run
        self._epoch_to_start = epoch
        self._config = config
        self._hybrid = config.get('hybrid_matching', False)
        self._hybrid_K = config.get('hybrid_K', 0)

        self._writer = SummaryWriter(log_dir=path_to_run)
        self._scaler = GradScaler()

        self._evaluator = DetectionEvaluator(
            classes=list(config['labels'].values()),
            classes_small=config['labels_small'],
            classes_mid=config['labels_mid'],
            classes_large=config['labels_large'],
            iou_range_nndet=(0.1, 0.5, 0.05),
            iou_range_coco=(0.5, 0.95, 0.05),
            sparse_results=True
        )

        # Init main metric for checkpoint
        self._main_metric_key = 'mAP_coco'
        self._main_metric_max_val = metric_start_val

    def _train_one_epoch(self, num_epoch):
        self._model.train()
        # self._criterion.train()

        loss_agg = 0
        loss_bbox_agg = 0
        loss_giou_agg = 0
        loss_cls_agg = 0
        loss_seg_ce_agg = 0
        loss_seg_dice_agg = 0
        loss_dn_agg = {}
        # Aux loss
        loss_aux_agg = {}
        loss_aux_agg_dn = {}
        # Two stage
        loss_enc_bbox_agg = 0
        loss_enc_giou_agg = 0
        loss_enc_cls_agg = 0
        # log Hausdorff
        hd95_agg = 0
        # contrastive loss
        loss_contrast_agg = 0
        # hybrid matching
        loss_bbox_one2many_agg = 0
        loss_giou_one2many_agg = 0
        loss_cls_one2many_agg = 0
        loss_seg_ce_one2many_agg = 0
        loss_seg_dice_one2many_agg = 0

        progress_bar = tqdm(self._train_loader)
        for data, _, bboxes, seg_targets in progress_bar:
            # Put data to gpu
            data, seg_targets = data.to(device=self._device), seg_targets.to(device=self._device)
        
            det_targets = []
            for item in bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': item[1].to(dtype=torch.long, device=self._device)
                }
                det_targets.append(target)

            # Make prediction
            with autocast(): 
                out, contrast_losses, dn_meta = self._model(data, det_targets, num_epoch=num_epoch)
                loss_dict, _ = self._criterion(out, det_targets, seg_targets, dn_meta, num_epoch=num_epoch)

                if self._criterion._seg_proxy: # log Hausdorff
                    hd95 = loss_dict['hd95'].item()
                del loss_dict['hd95'] # remove HD from loss, so it does not influence loss_abs

                if self._hybrid: # hybrid matching
                    outputs_one2many = dict()
                    outputs_one2many["pred_logits"] = out["pred_logits_one2many"]
                    outputs_one2many["pred_boxes"] = out["pred_boxes_one2many"]
                    outputs_one2many["aux_outputs"] = out["aux_outputs_one2many"]
                    outputs_one2many["seg_one2many"] = True
                    det_many_targets = copy.deepcopy(det_targets)
                    # repeat the targets
                    for target in det_many_targets:
                        target["boxes"] = target["boxes"].repeat(self._hybrid_K, 1)
                        target["labels"] = target["labels"].repeat(self._hybrid_K)

                    loss_dict_one2many, _ = self._criterion(outputs_one2many, det_many_targets, seg_targets)
                    del loss_dict_one2many['hd95']
                    for key, value in loss_dict_one2many.items():
                        if key + "_one2many" in loss_dict.keys():
                            loss_dict[key + "_one2many"] += value * self._config['hybrid_loss_weight_one2many']
                        else:
                            loss_dict[key + "_one2many"] = value * self._config['hybrid_loss_weight_one2many']


                # Create absolute loss and mult with loss coefficient
                loss_abs = 0
                for loss_key, loss_val in loss_dict.items():
                    loss_abs += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]
                for loss_key, loss_val in contrast_losses.items():
                    loss_abs += loss_val # already multiplied coefficient in transoarnet.py
                    loss_contrast_agg += loss_val 

            self._optimizer.zero_grad()
            self._scaler.scale(loss_abs).backward()

            # Clip grads to counter exploding grads
            max_norm = self._config['clip_max_norm']
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm)

            self._scaler.step(self._optimizer)
            self._scaler.update()

            loss_agg += loss_abs.item()
            loss_bbox_agg += loss_dict['bbox'].item()
            loss_giou_agg += loss_dict['giou'].item()
            loss_cls_agg += loss_dict['cls'].item()
            loss_seg_ce_agg += loss_dict['segce'].item()
            loss_seg_dice_agg += loss_dict['segdice'].item()
            if self._hybrid: # hybrid matching
                loss_bbox_one2many_agg += loss_dict['bbox_one2many'].item()
                loss_giou_one2many_agg += loss_dict['giou_one2many'].item()
                loss_cls_one2many_agg += loss_dict['cls_one2many'].item()
                loss_seg_ce_one2many_agg += loss_dict['segce_one2many'].item()
                loss_seg_dice_one2many_agg += loss_dict['segdice_one2many'].item()

            if self._criterion._seg_proxy: # log Hausdorff
                hd95_agg += hd95
            
            if dn_meta is not None:
                if len(loss_dn_agg) == 0: # initialize loss entries
                    loss_dn_agg['bbox_dn'] = 0
                    loss_dn_agg['giou_dn'] = 0
                    loss_dn_agg['cls_dn'] = 0
                else:
                    loss_dn_agg['bbox_dn'] += loss_dict[f'bbox_dn'].item()
                    loss_dn_agg['giou_dn'] += loss_dict[f'giou_dn'].item()
                    loss_dn_agg['cls_dn'] += loss_dict[f'cls_dn'].item()
            if "aux_outputs" in out:
                if len(loss_aux_agg) == 0: # initialize loss entries
                    for i in range(len(out["aux_outputs"])):
                        loss_aux_agg[f'bbox_{i}'] = 0
                        loss_aux_agg[f'giou_{i}'] = 0
                        loss_aux_agg[f'cls_{i}'] = 0
                for i in range(len(out["aux_outputs"])):
                    loss_aux_agg[f'bbox_{i}'] += loss_dict[f'bbox_{i}'].item()
                    loss_aux_agg[f'giou_{i}'] += loss_dict[f'giou_{i}'].item()
                    loss_aux_agg[f'cls_{i}'] += loss_dict[f'cls_{i}'].item()
                
                if dn_meta is not None:
                    if len(loss_aux_agg_dn) == 0: # initialize loss entries
                        for i in range(len(out["aux_outputs"])):
                            loss_aux_agg_dn[f'bbox_{i}_dn'] = 0
                            loss_aux_agg_dn[f'giou_{i}_dn'] = 0
                            loss_aux_agg_dn[f'cls_{i}_dn'] = 0
                    for i in range(len(out["aux_outputs"])):
                        loss_aux_agg_dn[f'bbox_{i}_dn'] += loss_dict[f'bbox_{i}_dn'].item()
                        loss_aux_agg_dn[f'giou_{i}_dn'] += loss_dict[f'giou_{i}_dn'].item()
                        loss_aux_agg_dn[f'cls_{i}_dn'] += loss_dict[f'cls_{i}_dn'].item()
            if "enc_outputs" in out:
                loss_enc_bbox_agg += loss_dict['bbox_enc'].item()
                loss_enc_giou_agg += loss_dict['giou_enc'].item()
                loss_enc_cls_agg += loss_dict['cls_enc'].item()
            memory_allocated, memory_cached = get_gpu_memory(self._device)
            progress_bar.set_postfix({'cached': "{:.2f}GB".format(memory_cached/(1024**3))})
            
        loss = loss_agg / len(self._train_loader)
        #print(f'total train loss for epoch {num_epoch}: '+str(loss))
        loss_bbox = loss_bbox_agg / len(self._train_loader)
        loss_giou = loss_giou_agg / len(self._train_loader)
        loss_cls = loss_cls_agg / len(self._train_loader)
        loss_seg_ce = loss_seg_ce_agg / len(self._train_loader)
        loss_seg_dice = loss_seg_dice_agg / len(self._train_loader)
        if self._hybrid: # hybrid matching
            loss_bbox_one2many = loss_bbox_one2many_agg / len(self._train_loader)
            loss_giou_one2many = loss_giou_one2many_agg / len(self._train_loader)
            loss_cls_one2many = loss_cls_one2many_agg / len(self._train_loader)
            loss_seg_ce_one2many = loss_seg_ce_one2many_agg / len(self._train_loader)
            loss_seg_dice_one2many = loss_seg_dice_one2many_agg / len(self._train_loader)
        
        if self._criterion._seg_proxy:  # log Hausdorff
            seg_hd95 = hd95_agg / len(self._train_loader)
        else:
            seg_hd95 = 0

        loss_contrast = loss_contrast_agg / len(self._train_loader)
        
        if len(loss_dn_agg) != 0:
            for key in loss_dn_agg:
                value = loss_dn_agg[key] / len(self._train_loader)
                self._writer.add_scalar("dn/"+key, value, num_epoch)
        if len(loss_aux_agg) != 0:
            for key in loss_aux_agg:
                value = loss_aux_agg[key] / len(self._train_loader)
                self._writer.add_scalar("train_aux/"+key, value, num_epoch)
        if len(loss_aux_agg_dn) != 0:
            for key in loss_aux_agg_dn:
                value = loss_aux_agg_dn[key] / len(self._train_loader)
                self._writer.add_scalar("dn/"+key, value, num_epoch)
        if loss_enc_bbox_agg or loss_enc_giou_agg or loss_enc_cls_agg:
            self._writer.add_scalar("train_enc/bbox_enc", loss_enc_bbox_agg/len(self._train_loader), num_epoch)
            self._writer.add_scalar("train_enc/giou_enc", loss_enc_giou_agg/len(self._train_loader), num_epoch)
            self._writer.add_scalar("train_enc/cls_enc", loss_enc_cls_agg/len(self._train_loader), num_epoch)

        if self._hybrid: # log many2one just if hybrid matching is activated
            self._write_to_logger(
                    num_epoch, 'train', 
                    total_loss=loss,
                    bbox_loss=loss_bbox,
                    giou_loss=loss_giou,
                    cls_loss=loss_cls,
                    seg_ce_loss=loss_seg_ce,
                    seg_dice_loss=loss_seg_dice,
                    seg_hd95=seg_hd95, # log Hausdorff
                    bbox_loss_one2many = loss_bbox_one2many,
                    giou_loss_one2many = loss_giou_one2many,
                    cls_loss_one2many = loss_cls_one2many,
                    seg_ce_loss_one2many = loss_seg_ce_one2many,
                    seg_dice_loss_one2many = loss_seg_dice_one2many
            )
        else:
            self._write_to_logger(
                num_epoch, 'train', 
                total_loss=loss,
                bbox_loss=loss_bbox,
                giou_loss=loss_giou,
                cls_loss=loss_cls,
                seg_ce_loss=loss_seg_ce,
                seg_dice_loss=loss_seg_dice,
                seg_hd95=seg_hd95, # log Hausdorff
                contrast_loss=loss_contrast,
            )

    @torch.no_grad()
    def _validate(self, num_epoch):
        self._model.eval()
        set_seg_proxy = self._model._seg_proxy
        set_msa_seg = self._model._msa_seg

        self._model._seg_proxy = False
        self._model._msa_seg = False
        self._criterion._seg_proxy = False
        self._criterion._seg_msa = False
        # self._criterion.eval()
        patches_count = 0

        loss_agg = 0
        loss_bbox_agg = 0
        loss_giou_agg = 0
        loss_cls_agg = 0
        loss_seg_ce_agg = 0
        loss_seg_dice_agg = 0
        # log Hausdorff
        hd95_agg = 0
        # Aux loss
        loss_aux_agg = {}
        # Two stage
        loss_enc_bbox_agg = 0
        loss_enc_giou_agg = 0
        loss_enc_cls_agg = 0
        progress_bar = tqdm(self._val_loader)
        for data, _, bboxes, seg_targets, patch_pos, padded_batch_labels, whole_padded_bboxes in progress_bar:
            # Put data to gpu
            inf_out_patches = {} # track predictions for all patches 
            for ch in range(data.shape[0]):
                patches_count += 1
                data_c, seg_targets_c = data[ch,:].to(device=self._device)[None,:], seg_targets[ch,:].to(device=self._device)[None,:]
                det_targets = []
                for item in [bboxes[ch]]:
                    target = {
                        'boxes': item[0].to(dtype=torch.float, device=self._device),
                        'labels': item[1].to(device=self._device)
                    }
                    det_targets.append(target)

                # Make prediction
                with autocast():
                    out = self._model(data_c)
                    loss_dict, _ = self._criterion(out, det_targets, seg_targets_c)

                    if self._criterion._seg_proxy: # log Hausdorff
                        hd95 = loss_dict['hd95'].item()
                    del loss_dict['hd95'] # remove HD from loss, so it does not influence loss_abs

                    # Create absolute loss and mult with loss coefficient
                    loss_abs = 0
                    for loss_key, loss_val in loss_dict.items():
                        loss_abs += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]
                # store patch-wisepredictions
                pred_boxes, pred_classes, pred_scores = inference(out)
                inf_out_patches[ch] = {"pred_boxes": pred_boxes[0],
                                       "pred_classes": pred_classes[0],
                                       "pred_scores": pred_scores[0]}
                # loss
                loss_agg += loss_abs.item()
                loss_bbox_agg += loss_dict['bbox'].item()
                loss_giou_agg += loss_dict['giou'].item()
                loss_cls_agg += loss_dict['cls'].item()
                loss_seg_ce_agg += loss_dict['segce'].item()
                loss_seg_dice_agg += loss_dict['segdice'].item()
                if self._criterion._seg_proxy: # log Hausdorff
                    hd95_agg += hd95

                
                if "aux_outputs" in out:
                    if len(loss_aux_agg) == 0: # initialize loss entries
                        for i in range(len(out["aux_outputs"])):
                            loss_aux_agg[f'bbox_{i}'] = 0
                            loss_aux_agg[f'giou_{i}'] = 0
                            loss_aux_agg[f'cls_{i}'] = 0
                    for i in range(len(out["aux_outputs"])):
                        loss_aux_agg[f'bbox_{i}'] += loss_dict[f'bbox_{i}'].item()
                        loss_aux_agg[f'giou_{i}'] += loss_dict[f'giou_{i}'].item()
                        loss_aux_agg[f'cls_{i}'] += loss_dict[f'cls_{i}'].item()
                    
                if "enc_outputs" in out:
                    loss_enc_bbox_agg += loss_dict['bbox_enc'].item()
                    loss_enc_giou_agg += loss_dict['giou_enc'].item()
                    loss_enc_cls_agg += loss_dict['cls_enc'].item()

            # Evaluate validation predictions based on metric
            pred_boxes, pred_classes, pred_scores = merge_patches(inf_out_patches, patch_pos, self._config['augmentation']['patch_size'], padded_batch_labels.shape[-3:], mode=self._config['patch_merge_mode'], config=self._config)
            # TODO add boxes for padded labels to use with eval

            targets = {
                        'boxes': whole_padded_bboxes[0][0].to(dtype=torch.float, device=self._device),
                        'labels': whole_padded_bboxes[0][1].to(device=self._device)
                    }
            
            self._evaluator.add(
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                pred_scores=pred_scores,
                gt_boxes=[targets['boxes'].detach().cpu().numpy()],
                gt_classes=[targets['labels'].detach().cpu().numpy()]
            )
            memory_allocated, memory_cached = get_gpu_memory(self._device)
            progress_bar.set_postfix({'cached': "{:.2f}GB".format(memory_cached/(1024**3))})

        loss = loss_agg / patches_count
        loss_bbox = loss_bbox_agg / patches_count
        loss_giou = loss_giou_agg / patches_count
        loss_cls = loss_cls_agg / patches_count
        loss_seg_ce = loss_seg_ce_agg / patches_count
        loss_seg_dice = loss_seg_dice_agg / patches_count
        if self._criterion._seg_proxy:  # log Hausdorff
            seg_hd95 = hd95_agg / patches_count
        else:
            seg_hd95 = 0

        metric_scores = self._evaluator.eval()
        self._evaluator.reset()

        # Check if new best checkpoint
        if metric_scores[self._main_metric_key] >= self._main_metric_max_val \
            and not self._config['debug_mode']:
            self._main_metric_max_val = metric_scores[self._main_metric_key]
            self._save_checkpoint(
                num_epoch,
                f'model_best_{metric_scores[self._main_metric_key]:.3f}_in_ep{num_epoch}.pt'
            )
            
        if len(loss_aux_agg) != 0:
            for key in loss_aux_agg:
                value = loss_aux_agg[key] / patches_count
                self._writer.add_scalar("val_aux/"+key, value, num_epoch)
        if loss_enc_bbox_agg or loss_enc_giou_agg or loss_enc_cls_agg:
            self._writer.add_scalar("val_enc/bbox_enc", loss_enc_bbox_agg/patches_count, num_epoch)
            self._writer.add_scalar("val_enc/giou_enc", loss_enc_giou_agg/patches_count, num_epoch)
            self._writer.add_scalar("val_enc/cls_enc", loss_enc_cls_agg/patches_count, num_epoch)

        # Write to logger
        self._write_to_logger(
            num_epoch, 'val', 
            total_loss=loss,
            bbox_loss=loss_bbox,
            giou_loss=loss_giou,
            cls_loss=loss_cls,
            seg_ce_loss=loss_seg_ce,
            seg_dice_loss=loss_seg_dice
        )

        self._write_to_logger(
            num_epoch, 'val_metric',
            mAPcoco=metric_scores['mAP_coco'],
            mAPcocoS=metric_scores['mAP_coco_s'],
            mAPcocoM=metric_scores['mAP_coco_m'],
            mAPcocoL=metric_scores['mAP_coco_l'],
            mAPnndet=metric_scores['mAP_nndet'],
            AP10=metric_scores['AP_IoU_0.10'],
            AP50=metric_scores['AP_IoU_0.50'],
            AP75=metric_scores['AP_IoU_0.75'],
            seg_hd95=seg_hd95 # log Hausdorff
        )
        self._model._seg_proxy = set_seg_proxy
        self._model._msa_seg = set_msa_seg
        self._criterion._seg_proxy = set_seg_proxy
        self._criterion._seg_msa = set_msa_seg

    def run(self):
        if self._epoch_to_start == 0:   # For initial performance estimation
            self._validate(0)

        for epoch in range(self._epoch_to_start + 1, self._config['epochs'] + 1):
            print("starting epoch ", epoch)
            self._train_one_epoch(epoch)

            # Log learning rates
            self._write_to_logger(
                epoch, 'lr',
                backbone=self._optimizer.param_groups[0]['lr'],
                neck=self._optimizer.param_groups[1]['lr']
            )

            if epoch % self._config['val_interval'] == 0:
                self._validate(epoch)

            self._scheduler.step()

            if not self._config['debug_mode']:
                self._save_checkpoint(epoch, 'model_last.pt')
            # fixed checkpoints at each 200 epochs:
            if (epoch % 500) == 0:
                self._save_checkpoint(epoch, f'model_epoch_{epoch}.pt')

    def _write_to_logger(self, num_epoch, category, **kwargs):
        for key, value in kwargs.items():
            name = category + '/' + key
            self._writer.add_scalar(name, value, num_epoch)

    def _save_checkpoint(self, num_epoch, name):
        # Delete prior best checkpoint
        if 'best' in name:
            [path.unlink() for path in self._path_to_run.iterdir() if 'best' in str(path)]

        torch.save({
            'epoch': num_epoch,
            'metric_max_val': self._main_metric_max_val,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict(),
        }, self._path_to_run / name)

def get_gpu_memory(device):
    #torch.cuda.empty_cache()
    memory_allocated = torch.cuda.memory_allocated(device)
    memory_cached = torch.cuda.memory_cached(device)
    return memory_allocated, memory_cached
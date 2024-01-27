"""Inference script to infer final output from model prediction."""

import numpy as np
import torch
import torch.nn.functional as F
from transoar.utils.bboxes import iou_3d_np

iou_checking = False  # prints ioUs of multiple predictions per class (e.g., for checking all predictions in dense matching)

def inference(out, vis_queries=False):
    # Get probabilities from output logits
    pred_probs = F.softmax(out['pred_logits'], dim=-1)

    # Transform into np arrays and store as a list of arrays, as required in evaluator
    pred_boxes = [boxes.detach().cpu().numpy() for boxes in out['pred_boxes']]
    pred_classes = [torch.max(probs, dim=-1)[1].detach().cpu().numpy() for probs in pred_probs]
    pred_scores = [torch.max(probs, dim=-1)[0].detach().cpu().numpy() for probs in pred_probs]

    #if type(out['pred_seg']) is not int or:
    #    pred_seg = F.softmax(out['pred_seg'], dim=1)
    #    pred_seg_map = [seg.detach().cpu().numpy() for seg in pred_seg]
    #else:
    #    pred_seg_map = 0

    # retun query ids for plotting sample locations
    quer = {}
    if vis_queries:
        uniq_cls = np.unique(pred_classes)
        for cls_ in uniq_cls:
            if cls_ == 0:
                continue
            indices = np.where(pred_classes[0] == cls_)
            quer[cls_] = indices

    # Get rid of empty detections
    valid_ids = [np.nonzero(batch_elem_classes) for batch_elem_classes in pred_classes]
    pred_classes = [pred[ids] for pred, ids in zip(pred_classes, valid_ids)]
    pred_boxes = [pred[ids] for pred, ids in zip(pred_boxes, valid_ids)]
    pred_scores = [pred[ids] for pred, ids in zip(pred_scores, valid_ids)]

    # Get detection with highest score for each class as final prediction
    for idx, (batch_boxes, batch_classes, batch_scores) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
        max_ids = []
        unique_classes = np.unique(batch_classes)
        min_iou = np.ones(len(unique_classes))
        median_iou = np.ones(len(unique_classes))
        for class_ in unique_classes:
            class_idx = (batch_classes == class_).nonzero()[0]

            if class_idx.size > 1:
                class_scores = batch_scores[class_idx]

                if iou_checking:
                    m_idx = np.nonzero(class_idx[class_scores > 0.1])  # score threshold of 0.1
                    class_bboxes = batch_boxes[class_idx]
                    class_bboxes = class_bboxes[m_idx]
                    
                    n = class_bboxes.shape[0]
                    iou_matrix = iou_3d_np(class_bboxes, class_bboxes)
                    min_iou[class_-1] = np.min(iou_matrix)
                    median_iou[class_-1] = np.median(iou_matrix)

                max_ids.append(class_idx[class_scores.argmax()])
            else:
                max_ids.append(class_idx.item())

        pred_classes[idx] = pred_classes[idx][max_ids]
        pred_scores[idx] = pred_scores[idx][max_ids]
        pred_boxes[idx] = pred_boxes[idx][max_ids]
        if iou_checking:
            np.set_printoptions(precision=2)
            np.set_printoptions(linewidth=150)
            print("min iou per class:    ", min_iou, "\nmedian iou per class: ", median_iou)
        # assert pred_classes[idx].size <= 20
    if vis_queries:
        return pred_boxes, pred_classes, pred_scores, quer    
    return pred_boxes, pred_classes, pred_scores

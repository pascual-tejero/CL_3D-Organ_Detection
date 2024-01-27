# ------------------------------------------------------------------------
# Modified from DINO (https://github.com/IDEA-Research/DINO/tree/main)
# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Modified from DN-DETR (https://github.com/IDEA-Research/DN-DETR/tree/main)
# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from torch.nn import functional as F

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def prepare_for_dn(dn_args, training, is_contrastive, num_queries, num_classes, hidden_dim, label_enc):
    """
    The major difference from DN-DAB-DETR is that the author process pattern embedding pattern embedding in its detector
    forward function and use learnable tgt embedding, so we change this function a little bit.
    :param dn_args: targets, scalar, label_noise_scale, box_noise_scale, num_patterns
    :param tgt_weight: use learnbal tgt in dab deformable detr
    :param embedweight: positional anchor queries
    :param batch_size: bs
    :param training: if it is training or inference
    :param num_queries: number of queires
    :param num_classes: number of classes
    :param hidden_dim: transformer hidden dim
    :param label_enc: encode labels in dn
    :return:
    """
    if training:
        targets, dn_number, multi_scale, label_noise_ratio, multiscale_label_noise_ratio_max, dn_box_noise_ratio, multi_scale_box_noise_ratio_max = dn_args
        
        if multi_scale:
            assert dn_number > 1, "dn_number should be larger than zero" 
            delta  = (multi_scale_box_noise_ratio_max - dn_box_noise_ratio) / (dn_number-1.)
            assert delta >= 0, "multi_scale_box_noise_ratio_max should be larger than dn_box_noise_ratio"
            box_noise_scale_list = [dn_box_noise_ratio+i*delta for i in range(dn_number)]
            
            delta  = (multiscale_label_noise_ratio_max - label_noise_ratio) / (dn_number-1.)
            assert delta >= 0, "multi_scale_box_noise_ratio_max should be larger than label_noise_ratio"
            label_noise_scale_list = [label_noise_ratio+i*delta for i in range(dn_number)]

        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        know_idx = [torch.nonzero(t) for t in known]
        known_num = [sum(k) for k in known]
        # you can uncomment this to use fix number of dn queries
        # if int(max(known_num))>0:
        #     scalar=scalar//int(max(known_num))

        # can be modified to selectively denosie some label or boxes; also known label prediction
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        # add noise
        known_indice = known_indice.repeat(dn_number, 1).view(-1)
        known_labels = labels.repeat(dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        # noise on the label
        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            if multi_scale:
                chosen_indice = []
                num_gts = torch.stack(known_num).sum()
                for i in range(len(box_noise_scale_list)):
                    # start and end idx of a single dn group
                    start, end = i * num_gts, (i+1) * num_gts
                    chosen_indice_tmp = torch.nonzero(p[start:end] < (label_noise_scale_list[i] * 0.5)).view(-1)
                    chosen_indice += (chosen_indice_tmp + i * num_gts)
                if len(chosen_indice) > 0:
                    chosen_indice = torch.stack(chosen_indice)
            else:
                chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            
            if len(chosen_indice) > 0:
                new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
        # noise on the box
        if dn_box_noise_ratio > 0:
            diff = torch.zeros_like(known_bbox_expand)
            diff[:, :3] = known_bboxs[:, 3:6] / 2
            diff[:, 3:6] = known_bboxs[:, 3:6] / 2
            if multi_scale:
                num_gts = torch.stack(known_num).sum()
                rand_part = torch.rand_like(known_bbox_expand)
                
                for i in range(len(box_noise_scale_list)):
                    # start and end idx of a single dn group
                    start, end = i * num_gts, (i+1) * num_gts
                    known_bbox_expand[start:end] += torch.mul((rand_part[start:end] * 2 - 1.0),
                                               diff[start:end]).cuda() * box_noise_scale_list[i]
                    known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)
            else:
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                               diff).cuda() * dn_box_noise_ratio
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)
                
        single_pad = int(max(known_num))
        pad_size = int(single_pad * dn_number)
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 6).cuda()
        m = known_labels_expaned.long().to('cuda')
        
        if is_contrastive: # pass one-hot label
            input_label_embed = F.one_hot(m, num_classes=num_classes+1).float()
            input_bbox_embed = known_bbox_expand
            padding_label = torch.zeros(pad_size, num_classes+1).cuda()
        else: # pass encoded label
            input_label_embed = label_enc(m) 
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)  
            padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1) 

        # map in order
        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
        
        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        dn_meta = {
            "pad_size": pad_size,
            "num_dn_group": dn_number,
        }
    else:  # no dn for inference
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    # input_query_label = input_query_label.transpose(0, 1)
    # input_query_bbox = input_query_bbox.transpose(0, 1)

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def prepare_for_cdn(dn_args, training, is_contrastive, num_queries, num_classes, hidden_dim, label_enc):
    """
    A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its
    detector forward function and use learnable tgt embedding, so we change this function a little bit.
    :param dn_args: targets, dn_number, label_noise_ratio, dn_box_noise_ratio, mtuli_scale (use multi scale noise or not)
    :param training: if it is training or inference
    :param num_queries: number of queires
    :param num_classes: number of classes
    :param hidden_dim: transformer hidden dim
    :param label_enc: encode labels in dn
    :return:
    """
    if training:
        targets, dn_number, multi_scale, label_noise_ratio, multiscale_label_noise_ratio_max, dn_box_noise_ratio, multi_scale_box_noise_ratio_max = dn_args
        
        if multi_scale:
            assert dn_number > 1, "dn_number should be larger than zero" 
            delta  = (multi_scale_box_noise_ratio_max - dn_box_noise_ratio) / (dn_number-1.)
            assert delta >= 0, "multi_scale_box_noise_ratio_max should be larger than dn_box_noise_ratio"
            box_noise_scale_list = [dn_box_noise_ratio+i*delta for i in range(dn_number)]
            
            delta  = (multiscale_label_noise_ratio_max - label_noise_ratio) / (dn_number-1.)
            assert delta >= 0, "multi_scale_box_noise_ratio_max should be larger than label_noise_ratio"
            label_noise_scale_list = [label_noise_ratio+i*delta for i in range(dn_number)]


        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat([torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox) # return nonzero index
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        # noisy labels
        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float()) 

            if multi_scale:
                chosen_indice = []
                num_gts = torch.stack(known_num).sum()
                for i in range(len(box_noise_scale_list)):
                    # start and end idx of a single dn group
                    start, end = 2 * i * num_gts, 2 * (i+1) * num_gts
                    chosen_indice_tmp = torch.nonzero(p[start:end] < (label_noise_scale_list[i] * 0.5)).view(-1)
                    chosen_indice += (chosen_indice_tmp + 2 * i * num_gts)
                chosen_indice = torch.stack(chosen_indice)
            else:
                chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            
            new_label = torch.randint_like(chosen_indice, 0, num_classes+1)  # randomly put a new one here as noised label
            known_labels_expaned.scatter_(0, chosen_indice, new_label)


            
        single_pad = int(max(known_num))
        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)

        # adding noise to bbox
        if dn_box_noise_ratio > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :3] = known_bboxs[:, :3] - known_bboxs[:, 3:6] / 2
            known_bbox_[:, 3:6] = known_bboxs[:, :3] + known_bboxs[:, 3:6] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :3] = known_bboxs[:, 3:6] / 2
            diff[:, 3:6] = known_bboxs[:, 3:6] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0

            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            
            if multi_scale:
                num_gts = torch.stack(known_num).sum()
                for i in range(len(box_noise_scale_list)):
                    # start and end idx of a single dn group
                    start, end = 2 * i * num_gts, 2 * (i+1) * num_gts
                    known_bbox_[start:end] = known_bbox_[start:end] + \
                        torch.mul(rand_part[start:end], diff[start:end]).cuda() * box_noise_scale_list[i]
                    known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
                    known_bbox_expand[start:end, :3] = (known_bbox_[start:end, :3] + known_bbox_[start:end, 3:6]) / 2
                    known_bbox_expand[start:end, 3:6] = known_bbox_[start:end, 3:6] - known_bbox_[start:end, :3]

            else:
                known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * dn_box_noise_ratio
                known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
                known_bbox_expand[:, :3] = (known_bbox_[:, :3] + known_bbox_[:, 3:6]) / 2
                known_bbox_expand[:, 3:6] = known_bbox_[:, 3:6] - known_bbox_[:, :3]


        m = known_labels_expaned.long().to("cuda")
        
        if is_contrastive: # pass one-hot label
            input_label_embed = F.one_hot(m, num_classes=num_classes+1).float()
            input_bbox_embed = known_bbox_expand
            padding_label = torch.zeros(pad_size, num_classes+1).cuda()
        else: # pass encoded label
            input_label_embed = label_enc(m) 
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)  
            padding_label = torch.zeros(pad_size, hidden_dim).cuda()
            
        padding_bbox = torch.zeros(pad_size, 6).cuda() 
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1) 


        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num]) 
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed


        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct GTs
        attn_mask[pad_size:, :pad_size] = True
        # gt cannot see queries
        attn_mask[:pad_size, pad_size:] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), single_pad * 2 * (i + 1) : pad_size] = True
            
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), single_pad * 2 * (i + 1) : pad_size] = True
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * 2 * i] = True

        dn_meta = {
            "pad_size": pad_size,
            "num_dn_group": dn_number,
        }
    else:
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None
    
    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
    post process of dn after output from the transformer
    put the dn part in the dn_meta
    """
    if dn_meta and dn_meta["pad_size"] > 0:
        output_known_class = outputs_class[:, :, : dn_meta["pad_size"], :]
        output_known_coord = outputs_coord[:, :, : dn_meta["pad_size"], :]
        outputs_class = outputs_class[:, :, dn_meta["pad_size"] :, :]
        outputs_coord = outputs_coord[:, :, dn_meta["pad_size"] :, :]
        out = {
            "pred_logits": output_known_class[-1],
            "pred_boxes": output_known_coord[-1],
        }
        if aux_loss:
            out["aux_outputs"] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta["output_known_lbs_bboxes"] = out
    return outputs_class, outputs_coord
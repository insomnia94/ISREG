from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json
import h5py
import time
from pprint import pprint
from PIL import Image, ImageDraw
import copy

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from Config import *


# IoU function
def computeIoU(box1, box2):
  # each box is of [x1, y1, w, h]
  inter_x1 = max(box1[0], box2[0])
  inter_y1 = max(box1[1], box2[1])
  inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
  inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

  if inter_x1 < inter_x2 and inter_y1 < inter_y2:
    inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
  else:
    inter = 0
  union = box1[2]*box1[3] + box2[2]*box2[3] - inter
  return float(inter)/union

def extract_visual_feat(inputs, resnet):
    output = resnet.conv1(inputs)
    output = resnet.bn1(output)
    output = resnet.relu(output)
    output = resnet.maxpool(output)
    output = resnet.layer1(output)
    output = resnet.layer2(output)
    output = resnet.layer3(output)
    output = resnet.layer4(output)
    output = resnet.avgpool(output)
    output = output.reshape(-1, 2048).detach()
    return output


def eval_split(loader, model, split, opt, normalization):

    verbose = opt.get('verbose', True)
    num_sents = opt.get('num_sents', -1)
    assert split != 'train', 'Check the evaluation split.'

    model.eval()

    loader.resetIterator(split)
    loss_sum = 0
    loss_evals = 0
    acc = 0
    predictions = []
    finish_flag = False
    model_time = 0

    while True:
        data = loader.getTestBatch(split, opt)
        att_weights = loader.get_attribute_weights()

        img_path = data['img_path']
        img_W = data['img_W']
        img_H = data['img_H']
        ref_ids = data['ref_ids']
        sent_ids = data['sent_ids']
        triad_feats = data['triad_feats']
        sent_raws = data['sent_raws']
        sent_triads = data['sent_triads']
        #sent_feats = data['sent_feats']
        triad_feats = data['triad_feats']
        gd_ixs = data['gd_ixs']
        gd_boxes = data['gd_boxes']

        # import the image
        img = Image.open(img_path)
        img = img.convert("RGB")

        for i, sent_id in enumerate(sent_ids):
            sent_raw = sent_raws[i]
            sent_triad = sent_triads[i]

            '''
            print("raw: ",end="")
            print(sent_raw)
            print("triad: ",end="")
            print(sent_triad)
            '''

            gd_box = data['gd_boxes'][i]

            history_actions = []
            for a_i in range(history_actions_length):
                history_actions.append(-1)

            box_x0 = 0
            box_y0 = 0
            box_x1 = img_W
            box_y1 = img_H
            box_w = box_x1 - box_x0
            box_h = box_y1 - box_y0

            o_box_x0, o_box_y0, o_box_x1, o_box_y1 = box_x0, box_y0, box_x1, box_y1

            show_img = copy.deepcopy(img)

            # loop here

            action_count = 0

            while True:
                # crop the target
                ref_img = img.crop((box_x0, box_y0, box_x1, box_y1))
                #ref_img.show()
                ref_imgs = []
                ref_imgs.append(normalization(ref_img).cpu().numpy())
                ref_imgs_tensor = torch.Tensor(ref_imgs).cuda()
                #ref_imgs_feat = model.extract_visual_feat(ref_imgs)

                location_tensor = torch.FloatTensor(np.array([float(box_x0)/float(img_W), float(box_y0)/float(img_H), float(box_x1)/float(img_W), float(box_y1)/float(img_H), (float(box_w)*float(box_h))/(img_W*img_H)]))
                location_tensor = location_tensor.view(1, -1).cuda()

                history_actions_tensor = torch.FloatTensor(np.array(history_actions))
                history_actions_tensor = history_actions_tensor.view(1, -1).cuda()

                actions_tensor, actions_cat = model(ref_imgs_tensor, triad_feats[i].unsqueeze(0), location_tensor, history_actions_tensor)
                action = int(torch.argmax(actions_tensor).cpu().numpy())

                history_actions.pop(0)
                history_actions.append(action)

                # excute the action and update the box_x0, box_y0, box_x1, box_y1

                # action
                # 0 up
                # 1 down
                # 2 left
                # 3 down
                # 4 stop

                if action_count > max_action_steps:
                    action = 4

                if action == 0:
                    box_y0 = int(box_y0 + move_ratio*(box_y1-box_y0))

                if action == 1:
                    box_y1 = int(box_y1 - move_ratio*(box_y1-box_y0))

                if action == 2:
                    box_x0 = int(box_x0 + move_ratio*(box_x1-box_x0))

                if action == 3:
                    box_x1 = int(box_x1 - move_ratio*(box_x1-box_x0))

                box_w = box_x1 - box_x0
                box_h = box_y1 - box_y0

                action_count += 1

                pred_box = [box_x0, box_y0, box_x1 - box_x0, box_y1 - box_y0]
                IoU = computeIoU(pred_box, gd_box)

                if (IoU > accuracy_thre) or (action_count > max_action_steps) or (action==4):
                    break

            pred_box = [box_x0, box_y0, box_x1-box_x0, box_y1-box_y0]

            gd_box = data['gd_boxes'][i]

            IoU = computeIoU(pred_box, gd_box)

            '''
            draw = ImageDraw.Draw(show_img)
            draw.rectangle((box_x0, box_y0, box_x1, box_y1), width=2)
            draw.rectangle((gd_box[0], gd_box[1], gd_box[0]+gd_box[2], gd_box[1]+gd_box[3]), width=5)
            show_img.show()

            print("IoU: ", end="")
            print(str(IoU))
            print()
            '''

            if IoU >= accuracy_thre:
                acc += 1

            loss_evals += 1

            entry = {}
            #entry['image_id'] = image_id
            entry['sent_id'] = sent_id
            entry['IoU'] = IoU

            predictions.append(entry)
            toc = time.time()

            if num_sents > 0  and loss_evals >= num_sents:
                finish_flag = True
                break

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']


        if verbose:
            print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%,  ' % \
                  (split, ix0, ix1, acc*100.0/loss_evals  ))

        model_time = 0

        if finish_flag or data['bounds']['wrapped']:
            break

    return acc / loss_evals




def eval_split_final(loader, model, split, opt, normalization, Re_model):

    verbose = opt.get('verbose', True)
    num_sents = opt.get('num_sents', -1)
    assert split != 'train', 'Check the evaluation split.'

    model.eval()

    loader.resetIterator(split)
    loss_sum = 0
    loss_evals = 0
    acc = 0
    predictions = []
    finish_flag = False
    model_time = 0

    sig = torch.nn.Sigmoid()

    while True:
        data = loader.getTestBatch(split, opt)
        att_weights = loader.get_attribute_weights()

        img_path = data['img_path']
        img_W = data['img_W']
        img_H = data['img_H']
        ref_ids = data['ref_ids']
        sent_ids = data['sent_ids']
        triad_feats = data['triad_feats']
        sent_raws = data['sent_raws']
        sent_triads = data['sent_triads']
        triad_feats = data['triad_feats']
        gd_ixs = data['gd_ixs']
        gd_boxes = data['gd_boxes']

        # import the image
        img = Image.open(img_path)
        img = img.convert("RGB")
        #img.show()

        for i, sent_id in enumerate(sent_ids):
            sent_raw = sent_raws[i]
            sent_triad = sent_triads[i]
            gd_box = data['gd_boxes'][i]
            history_actions = []
            for a_i in range(history_actions_length):
                history_actions.append(-1)
            box_x0 = 0
            box_y0 = 0
            box_x1 = img_W
            box_y1 = img_H
            box_w = box_x1 - box_x0
            box_h = box_y1 - box_y0
            o_box_x0, o_box_y0, o_box_x1, o_box_y1 = box_x0, box_y0, box_x1, box_y1
            show_img = copy.deepcopy(img)
            # loop here
            action_count = 0
            while True:
                # crop the target
                ref_img = img.crop((box_x0, box_y0, box_x1, box_y1))
                #ref_img.show()
                ref_imgs = []
                ref_imgs.append(normalization(ref_img).cpu().numpy())
                ref_imgs_tensor = torch.Tensor(ref_imgs).cuda()
                #ref_imgs_feat = model.extract_visual_feat(ref_imgs)
                location_tensor = torch.FloatTensor(np.array([float(box_x0)/float(img_W), float(box_y0)/float(img_H), float(box_x1)/float(img_W), float(box_y1)/float(img_H), (float(box_w)*float(box_h))/(img_W*img_H)]))
                location_tensor = location_tensor.view(1, -1).cuda()
                history_actions_tensor = torch.FloatTensor(np.array(history_actions))
                history_actions_tensor = history_actions_tensor.view(1, -1).cuda()
                actions_tensor, actions_cat = model(ref_imgs_tensor, triad_feats[i].unsqueeze(0), location_tensor, history_actions_tensor)
                action = int(torch.argmax(actions_tensor).cpu().numpy())
                history_actions.pop(0)
                history_actions.append(action)
                # excute the action and update the box_x0, box_y0, box_x1, box_y1
                # action
                # 0 up
                # 1 down
                # 2 left
                # 3 down
                # 4 stop
                if action_count > max_action_steps:
                    action = 4
                if action == 0:
                    box_y0 = int(box_y0 + move_ratio*(box_y1-box_y0))
                if action == 1:
                    box_y1 = int(box_y1 - move_ratio*(box_y1-box_y0))
                if action == 2:
                    box_x0 = int(box_x0 + move_ratio*(box_x1-box_x0))
                if action == 3:
                    box_x1 = int(box_x1 - move_ratio*(box_x1-box_x0))
                box_w = box_x1 - box_x0
                box_h = box_y1 - box_y0
                action_count += 1
                pred_box = [box_x0, box_y0, box_x1 - box_x0, box_y1 - box_y0]
                IoU = computeIoU(pred_box, gd_box)
                if (IoU > accuracy_thre) or (action_count > max_action_steps) or (action==4):
                    break
            pred_box = [box_x0, box_y0, box_x1-box_x0, box_y1-box_y0]
            Re_location_tensor = torch.FloatTensor(np.array(
                [float(box_x0) / float(img_W), float(box_y0) / float(img_H), float(box_x1) / float(img_W),
                 float(box_y1) / float(img_H)]))
            Re_location_tensor = Re_location_tensor.view(1, -1).cuda()
            Re_coordinate_tensor = Re_model(ref_imgs_tensor, triad_feats[i].unsqueeze(0), Re_location_tensor)
            Re_coordinate_tensor = sig(Re_coordinate_tensor).detach()
            Re_coordinate = Re_coordinate_tensor.cpu().numpy()
            Re_coordinate_wh = [Re_coordinate[0][0] * img_W, Re_coordinate[0][1] * img_H,
                                Re_coordinate[0][2] * img_W - Re_coordinate[0][0] * img_W,
                                Re_coordinate[0][3] * img_H - Re_coordinate[0][1] * img_H]
            gd_box = data['gd_boxes'][i]
            IoU = computeIoU(pred_box, gd_box)
            Re_Iou = computeIoU(Re_coordinate_wh, gd_box)
            print("raw: ", end="")
            print(sent_raw)
            print("triad: ", end="")
            print(sent_triad)
            draw = ImageDraw.Draw(show_img)
            draw.rectangle((box_x0, box_y0, box_x1, box_y1), width=2)
            draw.rectangle((gd_box[0], gd_box[1], gd_box[0]+gd_box[2], gd_box[1]+gd_box[3]), width=5)
            #show_img.show()
            print("IoU: ", end="")
            print(str(IoU))
            print()
            if IoU >= accuracy_thre:
                acc += 1
            elif (Re_Iou > 0.4):
                acc += 1
            loss_evals += 1
            entry = {}
            #entry['image_id'] = image_id
            entry['sent_id'] = sent_id
            entry['IoU'] = IoU
            predictions.append(entry)
            toc = time.time()
            if num_sents > 0  and loss_evals >= num_sents:
                finish_flag = True
                break
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if verbose:
            print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%,  ' % \
                  (split, ix0, ix1, acc*100.0/loss_evals  ))

        model_time = 0

        if finish_flag or data['bounds']['wrapped']:
            break

    return acc / loss_evals



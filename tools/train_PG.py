from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import json
import time
import random
from PIL import Image
from itertools import count
import sys
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# model
import _init_paths

from loaders.dataloader import DataLoader
from layers.model import Actor
from layers.model import Critic
import evals.utils as model_utils
import evals.eval as eval_utils
from opt import parse_opt
from Config import *

import torch

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

def load_model(checkpoint_path, actor_state_size, action_size):
    tic = time.time()
    model = Actor(actor_state_size, action_size)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'].state_dict())
    model.eval()
    model.cuda()
    print('model loaded in %.2f seconds' % (time.time() - tic))
    return model


def main(args):
    opt = vars(args)
    # initialize
    opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
    checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'], opt['exp_id'])
    if not osp.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    opt['actor_learning_rate'] = PG_actor_learning_rate
    opt['save_every'] = PG_save_every
    opt['learning_rate_decay_start'] = PG_learning_rate_decay_start
    opt['learning_rate_decay_every'] = PG_learning_rate_decay_every
    opt['max_iters'] = PG_max_iters
    opt['action_size'] = action_size
    opt['actor_state_size'] = actor_state_size
    opt['history_actions_length'] = history_actions_length
    opt['COCO_path'] = COCO_path


    # set random seed
    torch.manual_seed(opt['seed'])
    random.seed(opt['seed'])

    normalization = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if PG_first_train == True:
        actor = Actor(opt['actor_state_size'], opt['action_size'])
    else:
        model_prefix = osp.join('output', opt['dataset_splitBy'], opt['exp_id'], 'mrcn_cmr_with_st')
        infos = json.load(open(model_prefix + '.json'))
        model_opt = infos['opt']
        model_path = model_prefix + '.pth'
        actor = load_model(model_path, actor_state_size, action_size)





    critic = Critic(opt['actor_state_size'])

    # set up loader
    data_json = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.json')
    data_h5 = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.h5')
    loader = DataLoader(data_h5=data_h5, data_json=data_json, opt=opt, normalization=normalization)


    CE_loss = nn.CrossEntropyLoss(reduce=False)
    Softmax_loss = torch.nn.Softmax(dim=1)

    infos = {}
    if opt['start_from'] is not None:
        pass

    iter = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_accuracies = infos.get('val_accuracies', [])
    val_loss_history = infos.get('val_loss_history', {})
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    if opt['load_best_score'] == 1:
        best_val_score = infos.get('best_val_score', None)
    if opt['gpuid'] >= 0:
        actor.cuda()
        critic.cuda()

    actor_lr = opt['actor_learning_rate']

    # set up optimizer
    actor_optimizer = torch.optim.Adam(actor.parameters(),
                                 lr=actor_lr,
                                 betas=(opt['optim_alpha'], opt['optim_beta']),
                                 eps=opt['optim_epsilon'],
                                 weight_decay=opt['weight_decay'])


    data_time, model_time = 0, 0
    start_time = time.time()

    f = open("./result", "w")
    f.close()

    f = open("./PG_loss_log", "w")
    f.close()

    ref_img_tensor_pool = []
    sent_feat_pool = []
    location_tensor_pool = []
    history_actions_tensor_pool = []
    action_pool = []
    reward_pool = []

    steps = 0

    #acc = eval_utils.eval_split(loader, actor, 'val', opt, normalization)

    for e in count():

        #torch.cuda.empty_cache()

        T = {}
        tic = time.time()
        data = loader.getPGBatch('testA', opt)
        T['data'] = time.time() - tic
        tic = time.time()

        history_acc = 0
        current_acc = 0

        img_id = data['img_id']
        img_path = data['img_path']
        img_W = data['img_W']
        img_H = data['img_H']
        sent_feat = data['sent_feat']
        triad_feat = data['triad_feat']
        triad_raw = data['triad_raw']
        gd_box = data['gd_box']

        sent_feat = torch.Tensor(sent_feat).unsqueeze(0).cuda()
        triad_feat = torch.Tensor(triad_feat).unsqueeze(0).cuda()

        # import the image
        img = Image.open(img_path)
        img = img.convert("RGB")

        # the ground truth bounding box of the target
        gd_box_x0 = int(gd_box[0])
        gd_box_y0 = int(gd_box[1])
        gd_box_x1 = int(gd_box[2])
        gd_box_y1 = int(gd_box[3])
        gd_box_W = gd_box_x1 - gd_box_x0
        gd_box_H = gd_box_y1 - gd_box_y0
        gd_box_wh = [gd_box_x0, gd_box_y0, gd_box_W, gd_box_H]

        # current bounding box of the current search region
        t_box_x0 = 0
        t_box_y0 = 0
        t_box_x1 = img_W
        t_box_y1 = img_H
        t_box_w = t_box_x1 - t_box_x0
        t_box_h = t_box_y1 - t_box_y0

        # initialize the history_actions
        action = -1
        history_actions = []
        for a_i in range(opt['history_actions_length']):
            history_actions.append(-1)

        action_count = 0

        # run the epidemic
        for t in count():

            steps += 1

            t_box_wh = [int(t_box_x0), int(t_box_y0), int(t_box_x1 - t_box_x0), int(t_box_y1 - t_box_y0)]
            history_acc = computeIoU(gd_box_wh, t_box_wh)

            # state = ref_img_tensor + sent_feat + location_tensor + history_actions_tensor

            # ref_img_tensor
            ref_img = img.crop((t_box_x0, t_box_y0, t_box_x1, t_box_y1))
            #ref_img.show()
            ref_img = [normalization(ref_img).cpu().numpy()]
            ref_img_tensor = torch.Tensor(ref_img).cuda()

            # location_tensor
            location_tensor = torch.FloatTensor(np.array(
                [float(t_box_x0) / float(img_W), float(t_box_y0) / float(img_H), float(t_box_x1) / float(img_W),
                 float(t_box_y1) / float(img_H), (float(t_box_w) * float(t_box_h)) / (img_W * img_H)]))
            location_tensor = location_tensor.view(1, -1).cuda()

            # history_actions_tensor
            history_actions_tensor = torch.FloatTensor(np.array(history_actions))
            history_actions_tensor = history_actions_tensor.view(1, -1).cuda()

            # predict the action
            #actions_tensor, actions_cat = actor(ref_img_tensor, sent_feat, location_tensor, history_actions_tensor)
            actions_tensor, actions_cat = actor(ref_img_tensor, triad_feat, location_tensor, history_actions_tensor)

            action = actions_cat.sample()
            action_value = int(action.cpu().numpy())
            if action_count > max_action_steps:
                action_value = 4

            # update the history action list (add current action)
            history_actions.pop(0)
            history_actions.append(action_value)


            # execture the action
            # action
            # 0 up
            # 1 down
            # 2 left
            # 3 down
            # 4 stop

            if action_value == 0:
                t_box_y0 = int(t_box_y0 + move_ratio * (t_box_y1 - t_box_y0))

            if action_value == 1:
                t_box_y1 = int(t_box_y1 - move_ratio * (t_box_y1 - t_box_y0))

            if action_value == 2:
                t_box_x0 = int(t_box_x0 + move_ratio * (t_box_x1 - t_box_x0))

            if action_value == 3:
                t_box_x1 = int(t_box_x1 - move_ratio * (t_box_x1 - t_box_x0))

            t_box_w = t_box_x1 - t_box_x0
            t_box_h = t_box_y1 - t_box_y0



            # generate the reward (reward function)
            t_box_wh = [int(t_box_x0), int(t_box_y0), int(t_box_x1-t_box_x0), int(t_box_y1-t_box_y0)]
            IoU = computeIoU(gd_box_wh, t_box_wh)
            current_acc = IoU


            # discrete 1
            if IoU > 0.5:
                reward = 1
            else:
                reward = 0


            '''
            # discrete 2
            if IoU < 0.3:
                reward = 0
            elif IoU < 0.5:
                reward = 1
            else:
                reward = 10
            '''



            '''
            # continue
            if IoU < 0.5:
                reward = (IoU*IoU)*100
            else:
                reward = 100
            '''


            '''
            # difference
            if current_acc > 0.5:
                reward = 10
            elif current_acc > history_acc:
                reward = 1
            else:
                reward = 0
            '''


            # store related data for tranining
            ref_img_tensor_pool.append(ref_img_tensor)
            sent_feat_pool.append(sent_feat)
            location_tensor_pool.append(location_tensor)
            history_actions_tensor_pool.append(history_actions_tensor)
            action_pool.append(action)
            reward_pool.append(reward)

            # action to stop the episode
            if (action_value == 4) or (IoU>accuracy_thre) or (t>max_action_steps):
                break



        # update the policy
        if e > 0 and e % PG_e_batch == 0:
        #if e >= 0:

            # Discount reward
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * 0.99 + reward_pool[i]
                    reward_pool[i] = running_add

            # Normalize reward
            reward_mean = np.mean(reward_pool)
            #reward_std = np.std(reward_pool)
            for i in range(steps):
                #reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
                reward_pool[i] = reward_pool[i] - reward_mean

            # Gradient Desent
            actor.train()
            actor_optimizer.zero_grad()

            for i in range(steps):
                ref_img_tensor = ref_img_tensor_pool[i]
                sent_feat = sent_feat_pool[i]
                location_tensor = location_tensor_pool[i]
                history_actions_tensor = history_actions_tensor_pool[i]

                action = Variable(torch.FloatTensor([action_pool[i]])).cuda()
                reward = reward_pool[i]

                #actions_tensor, actions_cat = actor(ref_img_tensor, sent_feat, location_tensor, history_actions_tensor)
                actions_tensor, actions_cat = actor(ref_img_tensor, triad_feat, location_tensor, history_actions_tensor)

                log_prob = actions_cat.log_prob(action)

                loss = -log_prob * reward  # Negtive score function x reward
                loss.backward()

                actor_optimizer.step()

            ref_img_tensor_pool = []
            sent_feat_pool = []
            location_tensor_pool = []
            history_actions_tensor_pool = []
            action_pool = []
            reward_pool = []
            steps = 0


            T['model'] = time.time() - tic
            wrapped = data['bounds']['wrapped']

            data_time += T['data']
            model_time += T['model']

            total_time = (time.time() - start_time)/3600
            total_time = round(total_time, 2)


            if e % opt['losses_log_every'] == 0:
                print('e[%s], loss=%.3f, actor_lr=%.2E, time=%.3f h' % (e, loss.data[0].item(), actor_lr, total_time))
                data_time, model_time = 0, 0

                f = open("./PG_loss_log", "a")
                f.write(str('e[%s], loss=%.3f, actor_lr=%.2E, time=%.3f h' % (e, loss.data[0].item(), actor_lr, total_time)) + "\n")
                f.close()

            if opt['learning_rate_decay_start'] > 0 and e > opt['learning_rate_decay_start']:
                frac = (e - opt['learning_rate_decay_start']) / opt['learning_rate_decay_every']
                decay_factor = 0.1 ** frac
                actor_lr = opt['actor_learning_rate'] * decay_factor
                model_utils.set_lr(actor_optimizer, actor_lr)


            if (e % opt['save_every'] == 0) and (e > 0) or iter == opt['max_iters']:
            #if (e % opt['save_every'] == 0) or e == opt['max_iters']:

                acc = eval_utils.eval_split(loader, actor, 'testA', opt, normalization)
                val_accuracies += [(iter, acc)]
                print('validation acc : %.2f%%\n' % (acc * 100.0))

                current_score = acc

                f = open("./result", "a")
                f.write(str(current_score) + "\n")
                f.close()

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    checkpoint_path = osp.join(checkpoint_dir, opt['id'] + '.pth')
                    checkpoint = {}
                    checkpoint['model'] = actor
                    checkpoint['opt'] = opt
                    torch.save(checkpoint, checkpoint_path)
                    print('model saved to %s' % checkpoint_path)

                # write json report
                infos['e'] = e
                infos['iterators'] = loader.iterators
                infos['loss_history'] = loss_history
                infos['val_accuracies'] = val_accuracies
                infos['val_loss_history'] = val_loss_history
                infos['best_val_score'] = best_val_score

                infos['opt'] = opt
                infos['val_result_history'] = val_result_history

                #with open(osp.join(checkpoint_dir, opt['id'] + '.json'), 'w', encoding="utf8") as io:
                # json.dump(infos, io)
                with open(osp.join(checkpoint_dir, opt['id'] + '.json'), 'w') as io:
                    json.dump(infos, io)


            iter += 1
            if wrapped:
                epoch += 1
            if iter >= opt['max_iters'] and opt['max_iters'] > 0:
                print(str(best_val_score))
                break

if __name__ == '__main__':
    args = parse_opt()
    main(args)

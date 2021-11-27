from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import json
import time
import random
from PIL import Image
import sys
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
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

    opt['actor_learning_rate'] = SL_actor_learning_rate
    opt['save_every'] = SL_save_every
    opt['learning_rate_decay_start'] = SL_learning_rate_decay_start
    opt['learning_rate_decay_every'] = SL_learning_rate_decay_every
    opt['max_iters'] = SL_max_iters
    opt['action_size'] = action_size
    opt['actor_state_size'] = actor_state_size
    opt['history_actions_length'] = history_actions_length
    opt['batch_size'] = SL_batch_size
    opt['COCO_path'] = COCO_path


    # set random seed
    #torch.manual_seed(opt['seed'])
    #random.seed(opt['seed'])

    normalization = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if SL_first_train == True:
        actor = Actor(opt['actor_state_size'], opt['action_size'])
    else:
        model_prefix = osp.join('output', opt['dataset_splitBy'], opt['exp_id'], 'mrcn_cmr_with_st')
        infos = json.load(open(model_prefix + '.json'))
        model_opt = infos['opt']
        model_path = model_prefix + '.pth'
        actor = load_model(model_path, actor_state_size, action_size)

    # set up loader
    data_json = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.json')
    data_h5 = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.h5')
    loader = DataLoader(data_h5=data_h5, data_json=data_json, opt=opt, normalization=normalization)


    CE_loss = nn.CrossEntropyLoss(reduce=False)
    Softmax_loss = torch.nn.Softmax(dim=1)
    BCE_loss = nn.BCELoss()

    sig = torch.nn.Sigmoid()

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

    f = open("./SL_loss_log", "w")
    f.close()

    while True:

        #torch.cuda.empty_cache()

        actor.train()
        actor_optimizer.zero_grad()

        T = {}

        tic = time.time()
        data = loader.getSLBatch('train', opt)

        T['data'] = time.time() - tic
        tic = time.time()

        batch_ref_img_tensor = data['batch_ref_img_tensor']
        batch_sent_feat = data['batch_sent_feat']
        batch_triad_feat = data['batch_triad_feat']
        batch_action_tensor = data['batch_action_tensor']
        batch_location_tensor = data['batch_location_tensor']
        batch_history_actions_tensor = data['batch_history_actions_tensor']

        # pass into the actor model
        actions_tensor, actions_cat = actor(batch_ref_img_tensor, batch_triad_feat, batch_location_tensor, batch_history_actions_tensor)
        actions_tensor = sig(actions_tensor)

        #actor_loss = CE_loss(actions_tensor, batch_action_tensor)
        actor_loss = BCE_loss(actions_tensor, batch_action_tensor)
        actor_loss_sum = torch.sum(actor_loss)
        actor_loss_sum.backward()

        model_utils.clip_gradient(actor_optimizer, opt['grad_clip'])
        actor_optimizer.step()

        T['model'] = time.time() - tic
        wrapped = data['bounds']['wrapped']

        data_time += T['data']
        model_time += T['model']

        total_time = (time.time() - start_time)/3600
        total_time = round(total_time, 2)

        if iter % opt['losses_log_every'] == 0:
            loss_history[iter] = (actor_loss.data[0]).item()
            #print('i[%s], e[%s], sub_loss=%.1f, obj_loss=%.1f, rel_loss=%.1f, lr=%.2E, time=%.3f h' % (iter, epoch, sub_loss.data[0].item(), obj_loss.data[0].item(), rel_loss.data[0].item(), lr, total_time))
            print('i[%s], e[%s], loss=%.3f, actor_lr=%.2E, time=%.3f h' % (iter, epoch, actor_loss.data[0].item(), actor_lr, total_time))
            data_time, model_time = 0, 0

            f = open("./SL_loss_log", "a")
            f.write(str('i[%s], e[%s], loss=%.3f, actor_lr=%.2E, time=%.3f h' % (iter, epoch, actor_loss.data[0].item(), actor_lr, total_time)) + "\n")
            f.close()


        if opt['learning_rate_decay_start'] > 0 and iter > opt['learning_rate_decay_start']:
            frac = (iter - opt['learning_rate_decay_start']) / opt['learning_rate_decay_every']
            decay_factor = 0.1 ** frac
            actor_lr = opt['actor_learning_rate'] * decay_factor
            model_utils.set_lr(actor_optimizer, actor_lr)


        if (iter % opt['save_every'] == 0) and (iter > 0) or iter == opt['max_iters']:
        #if (iter % opt['eval_every'] == 0) or iter == opt['max_iters']:

            acc = eval_utils.eval_split(loader, actor, 'testA', opt, normalization)
            val_accuracies += [(iter, acc)]
            current_score = acc

            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                checkpoint_path = osp.join(checkpoint_dir, opt['id'] + '.pth')
                checkpoint = {}
                checkpoint['model'] = actor
                checkpoint['opt'] = opt
                torch.save(checkpoint, checkpoint_path)
                print('model saved to %s' % checkpoint_path)

            # write json report
            infos['iter'] = iter
            infos['epoch'] = epoch
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

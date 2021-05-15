"""
data_json has 
0. refs:       [{ref_id, ann_id, box, image_id, split, category_id, sent_ids, att_wds}]
1. images:     [{image_id, ref_ids, file_name, width, height, h5_id}]
2. anns:       [{ann_id, category_id, image_id, box, h5_id}]
3. sentences:  [{sent_id, tokens, h5_id}]
4. word_to_ix: {word: ix}
5. att_to_ix : {att_wd: ix}
6. att_to_cnt: {att_wd: cnt}
7. label_length: L

Note, box in [xywh] format
label_h5 has
/labels is (M, max_length) uint32 array of encoded labels, zeros padded
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import io

import os.path as osp
import numpy as np
import h5py
import json
import random
import cv2
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

import _init_paths
from loaders.loader import Loader
from mrcn import inference_no_imdb
import functools

from Config import *

# box functions
def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

class DataLoader(Loader):

    def __init__(self, data_json, data_h5, opt, normalization):
        # parent loader instance
        Loader.__init__(self, data_json, data_h5)

        self.normalization = normalization

        # prepare attributes
        self.att_to_ix = self.info['att_to_ix']
        self.ix_to_att = {ix: wd for wd, ix in self.att_to_ix.items()}
        self.num_atts = len(self.att_to_ix)
        self.att_to_cnt = self.info['att_to_cnt']

        # img_iterators for each split
        self.split_ix = {}
        self.iterators = {}
        for image_id, image in self.Images.items():
            # we use its ref's split (there is assumption that each image only has one split)
            split = self.Refs[image['ref_ids'][0]]['split']
            if split not in self.split_ix:
                self.split_ix[split] = []
                self.iterators[split] = 0
            self.split_ix[split] += [image_id]
        for k, v in self.split_ix.items():
            print('assigned %d images to split %s' %(len(v), k))

    def prepare_mrcn(self, head_feats_dir, args):
        """
        Arguments:
          head_feats_dir: cache/feats/dataset_splitBy/net_imdb_tag, containing all image conv_net feats
          args: imdb_name, net_name, iters, tag
        """
        self.head_feats_dir = head_feats_dir
        self.mrcn = inference_no_imdb.Inference(args)
        assert args.net_name == 'res101'
        self.pool5_dim = 1024
        self.fc7_dim = 2048

    # load different kinds of feats
    def loadFeats(self, Feats):
        # Feats = {feats_name: feats_path}
        self.feats = {}
        self.feat_dim = None
        for feats_name, feats_path in Feats.items():
            if osp.isfile(feats_path):
                self.feats[feats_name] = h5py.File(feats_path, 'r')
                self.feat_dim = self.feats[feats_name]['fc7'].shape[1]
                assert self.feat_dim == self.fc7_dim
                print('FeatLoader loading [%s] from %s [feat_dim %s]' %(feats_name, feats_path, self.feat_dim))

    # shuffle split
    def shuffle(self, split):
        random.shuffle(self.split_ix[split])

    # reset iterator
    def resetIterator(self, split):
        self.iterators[split]=0

    # expand list by seq per ref, i.e., [a,b], 3 -> [aaabbb]
    def expand_list(self, L, n):
        out = []
        for l in L:
            out += [l] * n
        return out

    def image_to_head(self, image_id):
        """Returns
        head: float32 (1, 1024, H, W)
        im_info: float32 [[im_h, im_w, im_scale]]
        """
        feats_h5 = osp.join(self.head_feats_dir, str(image_id)+'.h5')
        feats = h5py.File(feats_h5, 'r')
        head, im_info = feats['head'], feats['im_info']
        return np.array(head), np.array(im_info)

    def fetch_neighbour_ids(self, ann_id):
        """
        For a given ann_id, we return
        - st_ann_ids: same-type neighbouring ann_ids (not include itself)
        - dt_ann_ids: different-type neighbouring ann_ids
        Ordered by distance to the input ann_id
        """
        ann = self.Anns[ann_id]
        x,y,w,h = ann['box']
        rx, ry = x+w/2, y+h/2

        @functools.cmp_to_key
        def compare(ann_id0, ann_id1):
            x,y,w,h = self.Anns[ann_id0]['box']
            ax0, ay0 = x+w/2, y+h/2
            x,y,w,h = self.Anns[ann_id1]['box']
            ax1, ay1 = x+w/2, y+h/2
            # closer to farmer
            if (rx-ax0)**2+(ry-ay0)**2 <= (rx-ax1)**2+(ry-ay1)**2:
                return -1
            else:
                return 1

        image = self.Images[ann['image_id']]

        ann_ids = list(image['ann_ids'])
        ann_ids = sorted(ann_ids, key=compare)

        st_ann_ids, dt_ann_ids = [], []
        for ann_id_else in ann_ids:
            if ann_id_else != ann_id:
                if self.Anns[ann_id_else]['category_id'] == ann['category_id']:
                    st_ann_ids += [ann_id_else]
                else:
                    dt_ann_ids +=[ann_id_else]
        return st_ann_ids, dt_ann_ids

    def fetch_grid_feats(self, boxes, net_conv, im_info):
        """returns -pool5 (n, 1024, 7, 7) -fc7 (n, 2048, 7, 7)"""
        pool5, fc7 = self.mrcn.box_to_spatial_fc7(net_conv, im_info, boxes)
        return pool5, fc7

    def compute_lfeats(self, ann_ids):
        # return ndarray float32 (#ann_ids, 5)
        lfeats = np.empty((len(ann_ids), 5), dtype=np.float32)
        for ix, ann_id in enumerate(ann_ids):
            ann = self.Anns[ann_id]
            image = self.Images[ann['image_id']]
            x, y ,w, h = ann['box']
            ih, iw = image['height'], image['width']
            lfeats[ix] = np.array([x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)],np.float32)
        return lfeats

    def compute_dif_lfeats(self, ann_ids, topK=5):
        # return ndarray float32 (#ann_ids, 5*topK)
        dif_lfeats = np.zeros((len(ann_ids), 5*topK), dtype=np.float32)
        for i, ann_id in enumerate(ann_ids):
            # reference box
            rbox = self.Anns[ann_id]['box']
            rcx,rcy,rw,rh = rbox[0]+rbox[2]/2,rbox[1]+rbox[3]/2,rbox[2],rbox[3]
            st_ann_ids, _ =self.fetch_neighbour_ids(ann_id)
            # candidate box
            for j, cand_ann_id in enumerate(st_ann_ids[:topK]):
                cbox = self.Anns[cand_ann_id]['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                dif_lfeats[i, j*5:(j+1)*5] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        return dif_lfeats


    def fetch_attribute_label(self, ref_ids):
        """Return
    - labels    : Variable float (N, num_atts)
    - select_ixs: Variable long (n, )
    """
        labels = np.zeros((len(ref_ids), self.num_atts))
        select_ixs = []
        for i, ref_id in enumerate(ref_ids):
            # pdb.set_trace()
            ref = self.Refs[ref_id]
            if len(ref['att_wds']) > 0:
                select_ixs += [i]
                for wd in ref['att_wds']:
                    labels[i, self.att_to_ix[wd]] = 1

        return Variable(torch.from_numpy(labels).float().cuda()), Variable(torch.LongTensor(select_ixs).cuda())

    def fetch_cxt_feats(self, ann_ids, opt):
        """
        Return
        - cxt_feats : ndarray (#ann_ids, fc7_dim)
        - cxt_lfeats: ndarray (#ann_ids, ann_ids, 5)
        - dist: ndarray (#ann_ids, ann_ids, 1)
        Note we only use neighbouring "different" (+ "same") objects for computing context objects, zeros padded.
        """

        cxt_feats = np.zeros((len(ann_ids), self.fc7_dim), dtype=np.float32)
        cxt_lfeats = np.zeros((len(ann_ids), len(ann_ids), 5), dtype=np.float32)
        dist = np.zeros((len(ann_ids), len(ann_ids), 1), dtype=np.float32)

        for i, ann_id in enumerate(ann_ids):
            # reference box
            rbox = self.Anns[ann_id]['box']
            rcx, rcy, rw, rh = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
            for j, cand_ann_id in enumerate(ann_ids):
                cand_ann = self.Anns[cand_ann_id]
                # fc7_feats
                cxt_feats[i, :] = self.feats['ann']['fc7'][cand_ann['h5_id'], :]
                cbox = cand_ann['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                ccx, ccy = cbox[0]+cbox[2]/2, cbox[1]+cbox[3]/2
                dist[i,j,:] = np.array(abs(rcx-ccx)+abs(rcy-ccy))
                cxt_lfeats[i,j,:] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        return cxt_feats, cxt_lfeats, dist

    def extract_ann_features(self, image_id, opt):
        """Get features for all ann_ids in an image"""
        image = self.Images[image_id]
        ann_ids = image['ann_ids']

        # fetch image features
        head, im_info = self.image_to_head(image_id)
        head = Variable(torch.from_numpy(head).cuda())

        # fetch ann features
        ann_boxes = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in ann_ids]))
        ann_pool5, ann_fc7 = self.fetch_grid_feats(ann_boxes, head, im_info)

        # absolute location features
        lfeats = self.compute_lfeats(ann_ids)
        lfeats = Variable(torch.from_numpy(lfeats).cuda())

        # relative location features
        dif_lfeats = self.compute_dif_lfeats(ann_ids)
        dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())

        # fetch context_fc7 and context_lfeats
        cxt_fc7, cxt_lfeats, dist = self.fetch_cxt_feats(ann_ids, opt)
        cxt_fc7 = Variable(torch.from_numpy(cxt_fc7).cuda())
        cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())
        dist = Variable(torch.from_numpy(dist).cuda())

        return ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist


    def compute_rel_lfeats(self, sub_ann_id, obj_ann_id):
        if sub_ann_id == -1 or obj_ann_id == -1:
            rel_lfeats = torch.zeros(5)
        else:
            rbox = self.Anns[sub_ann_id]['box']
            rcx, rcy, rw, rh = rbox[0] + rbox[2] / 2, rbox[1] + rbox[3] / 2, rbox[2], rbox[3]
            cbox = self.Anns[obj_ann_id]['box']
            cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
            rel_lfeats = np.array(
                [(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx1 + cw - rcx) / rw, (cy1 + ch - rcy) / rh, cw * ch / (rw * rh)])
            rel_lfeats = torch.Tensor(rel_lfeats)
        return rel_lfeats

    # get batch of data
    def getSLBatch(self, split, opt):

        # options
        split_ix = self.split_ix[split]

        max_index = len(split_ix) - 1
        wrapped = False
        TopK = opt['num_cxt']
        dataset = opt["dataset"]
        splitBy = opt["splitBy"]

        # import the language feature of each sentence extracted through BERT
        sent_feats_path = 'cache/prepro/' + str(dataset) + "_" + str(splitBy) + "/refer_sents_bert.npy"
        sent_feats = np.load(sent_feats_path)

        # import the language feature of each triad
        sent_extract = json.load(open(osp.join('cache/sub_obj_wds', str(dataset) + '_' + str(splitBy), "sent_extract_multi.json")))

        sent_sub_wordid = sent_extract["sent_sub_wordid"]
        sent_sub_classwordid = sent_extract["sent_sub_classwordid"]
        sent_obj_wordid = sent_extract["sent_obj_wordid"]
        sent_rel_wordid = sent_extract["sent_rel_wordid"]

        # import the language feature of each word (word2vec)
        embedmat_path = 'cache/word_embedding/embed_matrix.npy'
        embedding_mat = np.load(embedmat_path)

        # import the vocabulary
        vocab_file = 'cache/word_embedding/vocabulary_72700.txt'
        f = open(vocab_file, "r")
        vocab_list = f.read().splitlines()
        f.close()


        # iterator using global parameter self.iterator
        ri = self.iterators[split]
        ri_next = ri+1
        if ri_next > max_index:
            ri_next = 0
            wrapped = True
        self.iterators[split] = ri_next

        batch_ref_img = []
        batch_action = []
        batch_location = []
        batch_history_actions = []
        batch_sent_feat = []
        batch_triad_feat = []
        batch_sent_tokens = []

        history_actions = []
        for a_i in range(opt['history_actions_length']):
            history_actions.append(-1)

        for i in range(opt['batch_size']):
            # access the current image
            img_id = split_ix[random.randint(0, len(split_ix)-1)]
            # note that some anns are not used (not referred)
            ann_ids = self.Images[img_id]['ann_ids']
            ref_ids = self.Images[img_id]['ref_ids']

            # generate the filename, W, H and the original anns, refs of this image
            img_W = self.Images[img_id]['width']
            img_H = self.Images[img_id]['height']
            img_filename = str(self.Images[img_id]['file_name'])
            img_path = opt['COCO_path'] + str(img_filename)
            img = Image.open(img_path)
            img = img.convert("RGB")
            #img.show()

            ref_id = ref_ids[random.randint(0, len(ref_ids) - 1)]
            ref = self.Refs[ref_id]

            distract_ref_id = ref_ids[random.randint(0, len(ref_ids) - 1)]
            distract_ref = self.Refs[distract_ref_id]

            sent_ids = ref['sent_ids']
            sent_id = sent_ids[random.randint(0, len(sent_ids) - 1)]
            sent_tokens = self.Sentences[sent_id]['tokens']
            #print(sent_tokens)
            sent_feat = sent_feats[sent_id]


            # first triad
            sub_classwordid = sent_sub_classwordid[str(sent_id)][0]
            obj_wordid = sent_obj_wordid[str(sent_id)][0]
            rel_wordid = sent_rel_wordid[str(sent_id)][0]

            sub_classword = vocab_list[sub_classwordid]
            obj_word = vocab_list[obj_wordid]
            rel_word = vocab_list[rel_wordid]

            sub_classword_feat_np = embedding_mat[sub_classwordid]
            obj_word_feat_np = embedding_mat[obj_wordid]
            rel_word_feat_np = embedding_mat[rel_wordid]

            # second triad
            if len(sent_sub_classwordid[str(sent_id)]) > 1:
                sub_classwordid_2 = sent_sub_classwordid[str(sent_id)][1]
                obj_wordid_2 = sent_obj_wordid[str(sent_id)][1]
                rel_wordid_2 = sent_rel_wordid[str(sent_id)][1]

                sub_classword_2 = vocab_list[sub_classwordid_2]
                obj_word_2 = vocab_list[obj_wordid_2]
                rel_word_2 = vocab_list[rel_wordid_2]

                sub_classword_feat_np_2 = embedding_mat[sub_classwordid_2]
                obj_word_feat_np_2 = embedding_mat[obj_wordid_2]
                rel_word_feat_np_2 = embedding_mat[rel_wordid_2]
            else:

                sub_classwordid_2 = sub_classwordid
                obj_wordid_2 = obj_wordid
                rel_wordid_2 = rel_wordid

                sub_classword_2 = sub_classword
                obj_word_2 = obj_word
                rel_word_2 = rel_word

                sub_classword_feat_np_2 = sub_classword_feat_np
                obj_word_feat_np_2 = obj_word_feat_np
                rel_word_feat_np_2 = rel_word_feat_np

            #triad_feat_np = np.concatenate((sub_classword_feat_np,obj_word_feat_np,rel_word_feat_np), axis=0)
            triad_feat_np = np.concatenate((sub_classword_feat_np, obj_word_feat_np, rel_word_feat_np,
                                            sub_classword_feat_np_2, obj_word_feat_np_2, rel_word_feat_np_2), axis=0)

            #triad = sub_classword + " " + rel_word + " " + obj_word
            triad = sub_classword + " " + rel_word + " " + obj_word + ", " + sub_classword_2 + " " + obj_word_2 + " " + rel_word_2


            gd_box = ref['box']
            box_x0 = int(gd_box[0])
            box_y0 = int(gd_box[1])
            box_x1 = int(gd_box[0]) + int(gd_box[2])
            box_y1 = int(gd_box[1]) + int(gd_box[3])
            box_w = box_x1 - box_x0
            box_h = box_y1 - box_y0

            # original coordinates for visualization
            o_box_x0, o_box_y0, o_box_x1, o_box_y1 = box_x0, box_y0, box_x1, box_y1


            # major_action 0: extend, 1: stop
            extend_prob = random.uniform(0, 1)

            # generate the sample action
            action_list = [0, 0, 0, 0, 0]


            # decide the specific action list
            # [0] top side
            # [1] bottom side
            # [2] left side
            # [3] right side
            # [4] stop

            if extend_prob > extend_prob_thre:

                # initialize the expand action for each direction (0:not expand, 1:expand)

                if random.uniform(0,1) > direction_thre:
                    expand_action_0 = 1
                else:
                    expand_action_0 = 0

                if random.uniform(0,1) > direction_thre:
                    expand_action_1 = 1
                else:
                    expand_action_1 = 0

                if random.uniform(0,1) > direction_thre:
                    expand_action_2 = 1
                else:
                    expand_action_2 = 0

                if random.uniform(0,1) > direction_thre:
                    expand_action_3 = 1
                else:
                    expand_action_3 = 0




                # decide whether a direction is suitable for extending
                if (expand_action_0==1) and ((box_y0/box_h)<unsuitable_thre):
                    expand_action_0 = 0

                if (expand_action_1==1) and (((img_H-box_y1)/box_h)<unsuitable_thre):
                    expand_action_1 = 0

                if (expand_action_2==1) and ((box_x0/box_w)<unsuitable_thre):
                    expand_action_2 = 0

                if (expand_action_3==1) and (((img_W-box_x1)/box_w)<unsuitable_thre):
                    expand_action_3 = 0




                action_list[0] = expand_action_0
                action_list[1] = expand_action_1
                action_list[2] = expand_action_2
                action_list[3] = expand_action_3

                if 1 not in action_list:
                    action_list[4] = 1

            if extend_prob <= extend_prob_thre:
                action_list[4] = 1



            # conduct the extending process

            if action_list[4] == 0:

                # randomly, slightly expand the bounding box
                expand_ration = random.uniform(expand_thre_A, expand_thre_B)
                box_y0 = int(expand_ration * box_y0)
                box_y1 = int(box_y1 + (1-expand_ration) * (img_H - box_y1))
                box_x0 = int(expand_ration * box_x0)
                box_x1 = int(box_x1 + (1-expand_ration) * (img_W - box_x1))

                # extend the bounding to particular directions
                extend_ration = random.uniform(extend_thre_A, extend_thre_B)

                if action_list[0] == 1:
                    box_y0 = int(extend_ration * box_y0)

                if action_list[1] == 1:
                    box_y1 = int(box_y1 + (1-extend_ration) * (img_H - box_y1))

                if action_list[2] == 1:
                    box_x0 = int(extend_ration * box_x0)

                if action_list[3] == 1:
                    box_x1 = int(box_x1 + (1-extend_ration) * (img_W - box_x1))

            '''
            # visualization for test only
            print(triad)
            if action_list[0]==1: print("top")
            if action_list[1]==1: print("bottom")
            if action_list[2]==1: print("left")
            if action_list[3]==1: print("right")
            if action_list[4]==1: print("stop")
            print()

            draw = ImageDraw.Draw(img)
            draw.rectangle((o_box_x0, o_box_y0, o_box_x1, o_box_y1), width=3)
            draw.rectangle((box_x0, box_y0, box_x1, box_y1), width=5)
            img.show()
            '''


            ref_img = img.crop((box_x0, box_y0, box_x1, box_y1))
            ref_img = self.normalization(ref_img).cpu().numpy()

            location = np.array([float(box_x0)/float(img_W), float(box_y0)/float(img_H), float(box_x1)/float(img_W), float(box_y1)/float(img_H), (float(box_w)*float(box_h))/(img_W*img_H)])

            batch_ref_img.append(ref_img)
            batch_action.append(action_list)
            batch_location.append(location)
            batch_history_actions.append(history_actions)
            batch_sent_feat.append(sent_feat)
            batch_sent_tokens.append(sent_tokens)
            batch_triad_feat.append(triad_feat_np)

        batch_ref_img_tensor = torch.Tensor(batch_ref_img).cuda()
        batch_sent_feat = torch.Tensor(batch_sent_feat).cuda()
        batch_triad_feat = torch.Tensor(batch_triad_feat).cuda()
        batch_action_tensor = torch.FloatTensor(np.array(batch_action)).cuda()
        batch_location_tensor = torch.FloatTensor(np.array(batch_location)).cuda()
        batch_history_actions_tensor = torch.FloatTensor(np.array(batch_history_actions)).cuda()



        data = {}

        data['batch_ref_img_tensor'] = batch_ref_img_tensor
        data['batch_sent_feat'] = batch_sent_feat
        data['batch_triad_feat'] = batch_triad_feat
        data['batch_action_tensor'] = batch_action_tensor
        data['batch_location_tensor'] = batch_location_tensor
        data['batch_history_actions_tensor'] = batch_history_actions_tensor
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}

        return data





    def getPGBatch(self, split, opt):

        # options
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        wrapped = False
        dataset = opt["dataset"]
        splitBy = opt["splitBy"]

        # import the language feature of each triad
        sent_extract = json.load(
            open(osp.join('cache/sub_obj_wds', str(dataset) + '_' + str(splitBy), "sent_extract_multi.json")))

        sent_sub_wordid = sent_extract["sent_sub_wordid"]
        sent_sub_classwordid = sent_extract["sent_sub_classwordid"]
        sent_obj_wordid = sent_extract["sent_obj_wordid"]
        sent_rel_wordid = sent_extract["sent_rel_wordid"]

        # import the language feature of each word (word2vec)
        embedmat_path = 'cache/word_embedding/embed_matrix.npy'
        embedding_mat = np.load(embedmat_path)

        # import the vocabulary
        vocab_file = 'cache/word_embedding/vocabulary_72700.txt'
        f = open(vocab_file, "r")
        vocab_list = f.read().splitlines()
        f.close()



        # iterator using global parameter self.iterator
        ri = self.iterators[split]
        ri_next = ri+1
        if ri_next > max_index:
            ri_next = 0
            wrapped = True
        self.iterators[split] = ri_next

        # access the current image
        img_id = split_ix[random.randint(0, len(split_ix)-1)]
        # note that some anns are not used (not referred)
        ann_ids = self.Images[img_id]['ann_ids']
        ref_ids = self.Images[img_id]['ref_ids']

        # generate the filename, W, H and the original anns, refs of this image
        img_W = self.Images[img_id]['width']
        img_H = self.Images[img_id]['height']
        img_filename = str(self.Images[img_id]['file_name'])
        img_path = opt['COCO_path'] + str(img_filename)

        ref_id = ref_ids[random.randint(0, len(ref_ids) - 1)]
        ref = self.Refs[ref_id]

        sent_ids = ref['sent_ids']
        sent_id = sent_ids[random.randint(0, len(sent_ids) - 1)]
        sent_tokens = self.Sentences[sent_id]['tokens']
        #print(sent_tokens)
        #sent_feat = sent_feats[sent_id]

        # first triad
        sub_classwordid = sent_sub_classwordid[str(sent_id)][0]
        obj_wordid = sent_obj_wordid[str(sent_id)][0]
        rel_wordid = sent_rel_wordid[str(sent_id)][0]

        sub_classword = vocab_list[sub_classwordid]
        obj_word = vocab_list[obj_wordid]
        rel_word = vocab_list[rel_wordid]

        sub_classword_feat_np = embedding_mat[sub_classwordid]
        obj_word_feat_np = embedding_mat[obj_wordid]
        rel_word_feat_np = embedding_mat[rel_wordid]

        # second triad
        if len(sent_sub_classwordid[str(sent_id)]) > 1:
            sub_classwordid_2 = sent_sub_classwordid[str(sent_id)][1]
            obj_wordid_2 = sent_obj_wordid[str(sent_id)][1]
            rel_wordid_2 = sent_rel_wordid[str(sent_id)][1]

            sub_classword_2 = vocab_list[sub_classwordid_2]
            obj_word_2 = vocab_list[obj_wordid_2]
            rel_word_2 = vocab_list[rel_wordid_2]

            sub_classword_feat_np_2 = embedding_mat[sub_classwordid_2]
            obj_word_feat_np_2 = embedding_mat[obj_wordid_2]
            rel_word_feat_np_2 = embedding_mat[rel_wordid_2]
        else:

            sub_classwordid_2 = sub_classwordid
            obj_wordid_2 = obj_wordid
            rel_wordid_2 = rel_wordid

            sub_classword_2 = sub_classword
            obj_word_2 = obj_word
            rel_word_2 = rel_word

            sub_classword_feat_np_2 = sub_classword_feat_np
            obj_word_feat_np_2 = obj_word_feat_np
            rel_word_feat_np_2 = rel_word_feat_np

        # triad_feat_np = np.concatenate((sub_classword_feat_np,obj_word_feat_np,rel_word_feat_np), axis=0)
        triad_feat_np = np.concatenate((sub_classword_feat_np, obj_word_feat_np, rel_word_feat_np,
                                        sub_classword_feat_np_2, obj_word_feat_np_2, rel_word_feat_np_2), axis=0)

        # triad = sub_classword + " " + rel_word + " " + obj_word
        triad = sub_classword + " " + rel_word + " " + obj_word + ", " + sub_classword_2 + " " + rel_word_2 + " " + obj_word_2





        gd_box = ref['box']
        box_x0 = int(gd_box[0])
        box_y0 = int(gd_box[1])
        box_x1 = int(gd_box[0]) + int(gd_box[2])
        box_y1 = int(gd_box[1]) + int(gd_box[3])

        gd_box = [float(box_x0), float(box_y0), float(box_x1), float(box_y1)]

        data = {}

        data['img_id'] = img_id
        data['img_path'] = img_path
        data['img_W'] = img_W
        data['img_H'] = img_H
        data['triad_feat'] = triad_feat_np
        data['triad_raw'] = triad
        data['gd_box'] = gd_box
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}

        return data



    def get_attribute_weights(self, scale = 10):
        # weights = \lamda * 1/sqrt(cnt)
        cnts = [self.att_to_cnt[self.ix_to_att[ix]] for ix in range(self.num_atts)]
        cnts = np.array(cnts)
        weights = 1 / cnts ** 0.5
        weights = (weights-np.min(weights))/(np.max(weights)-np.min(weights))
        weights = weights * (scale - 1) + 1
        return torch.from_numpy(weights).float()

    def decode_attribute_label(self, scores):
        """- scores: Variable (cuda) (n, num_atts) after sigmoid range [0, 1]
           - labels:list of [[att, sc], [att, sc], ...
        """
        scores = scores.data.cpu().numpy()
        N = scores.shape[0]
        labels = []
        for i in range(N):
            label = []
            score = scores[i]
            for j, sc in enumerate(list(score)):
                label += [(self.ix_to_att[j], sc)]
                labels.append(label)
        return labels




    def getTestBatch(self, split, opt):


        dataset = opt["dataset"]
        splitBy = opt["splitBy"]

        # import the language feature of each triad
        sent_extract = json.load(
            open(osp.join('cache/sub_obj_wds', str(dataset) + '_' + str(splitBy), "sent_extract_multi.json")))
        sent_sub_wordid = sent_extract["sent_sub_wordid"]
        sent_sub_classwordid = sent_extract["sent_sub_classwordid"]
        sent_obj_wordid = sent_extract["sent_obj_wordid"]
        sent_rel_wordid = sent_extract["sent_rel_wordid"]

        # import the language feature of each word (word2vec)
        embedmat_path = 'cache/word_embedding/embed_matrix.npy'
        embedding_mat = np.load(embedmat_path)

        # import the vocabulary
        vocab_file = 'cache/word_embedding/vocabulary_72700.txt'
        f = open(vocab_file, "r")
        vocab_list = f.read().splitlines()
        f.close()



        wrapped = False

        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        ri = self.iterators[split]
        ri_next = ri + 1

        if ri_next > max_index:
            ri_next = 0
            wrapped = True

        self.iterators[split] = ri_next

        image_id = split_ix[ri]
        image = self.Images[image_id]

        # generate the filename, W, H and the original anns, refs of this image
        img_W = self.Images[image_id]['width']
        img_H = self.Images[image_id]['height']
        img_filename = str(self.Images[image_id]['file_name'])
        img_path = COCO_path + str(img_filename)

        ann_ids = image['ann_ids']
        ref_ids = self.Images[image_id]['ref_ids']

        sent_ids = []
        img_sent_feats = []
        img_triad_feats = []
        sent_raws = []
        sent_triads = []
        gd_ixs = []
        gd_boxes = []
        att_refs = []

        for ref_id in image['ref_ids']:
            ref = self.Refs[ref_id]
            for sent_id in ref['sent_ids']:
                sent_ids += [sent_id]
                sent_token_list = self.Sentences[sent_id]['tokens']
                sent_raw = ""
                for token in sent_token_list:
                    sent_raw += token
                    sent_raw += " "
                sent_raws += [sent_raw]

                # generate the triad feature

                # first triad
                sub_classwordid = sent_sub_classwordid[str(sent_id)][0]
                obj_wordid = sent_obj_wordid[str(sent_id)][0]
                rel_wordid = sent_rel_wordid[str(sent_id)][0]

                sub_classword = vocab_list[sub_classwordid]
                obj_word = vocab_list[obj_wordid]
                rel_word = vocab_list[rel_wordid]

                sub_classword_feat_np = embedding_mat[sub_classwordid]
                obj_word_feat_np = embedding_mat[obj_wordid]
                rel_word_feat_np = embedding_mat[rel_wordid]

                # second triad
                if len(sent_sub_classwordid[str(sent_id)]) > 1:
                    sub_classwordid_2 = sent_sub_classwordid[str(sent_id)][1]
                    obj_wordid_2 = sent_obj_wordid[str(sent_id)][1]
                    rel_wordid_2 = sent_rel_wordid[str(sent_id)][1]

                    sub_classword_2 = vocab_list[sub_classwordid_2]
                    obj_word_2 = vocab_list[obj_wordid_2]
                    rel_word_2 = vocab_list[rel_wordid_2]

                    sub_classword_feat_np_2 = embedding_mat[sub_classwordid_2]
                    obj_word_feat_np_2 = embedding_mat[obj_wordid_2]
                    rel_word_feat_np_2 = embedding_mat[rel_wordid_2]
                else:

                    sub_classwordid_2 = sub_classwordid
                    obj_wordid_2 = obj_wordid
                    rel_wordid_2 = rel_wordid

                    sub_classword_2 = sub_classword
                    obj_word_2 = obj_word
                    rel_word_2 = rel_word

                    sub_classword_feat_np_2 = sub_classword_feat_np
                    obj_word_feat_np_2 = obj_word_feat_np
                    rel_word_feat_np_2 = rel_word_feat_np


                triad_feat_np = np.concatenate((sub_classword_feat_np, obj_word_feat_np, rel_word_feat_np, sub_classword_feat_np_2, obj_word_feat_np_2, rel_word_feat_np_2), axis=0)
                img_triad_feats += [triad_feat_np]

                triad = sub_classword + " " + rel_word + " " + obj_word + ", " + sub_classword_2 + " " + obj_word_2 + " " + rel_word_2
                sent_triads += [triad]

                gd_ixs += [ann_ids.index(ref['ann_id'])]
                gd_boxes += [ref['box']]
                att_refs += [ref_id]


        data = {}

        data['image_id'] = image_id
        data['img_path'] = img_path
        data['img_W'] = img_W
        data['img_H'] = img_H
        data['ref_ids'] = ref_ids
        data['sent_ids'] = sent_ids
        data['triad_feats'] = torch.Tensor(np.array(img_triad_feats)).cuda()
        data['gd_ixs'] = gd_ixs
        data['sent_raws'] = sent_raws
        data['sent_triads'] = sent_triads
        data['gd_boxes'] = gd_boxes
        data['ann_ids'] = ann_ids
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}

        return data


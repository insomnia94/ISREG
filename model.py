from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import datetime

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torchvision.models as models

def pad_out(k):
    "padding to have same size"
    return (k-1)//2


class FPN_backbone(nn.Module):
    def __init__(self, c3_ch, c4_ch, c5_ch, feat_size=256):
        super(FPN_backbone, self).__init__()

        self.feat_size = feat_size

        self.P7_2 = nn.Conv2d(in_channels=self.feat_size,
                              out_channels=self.feat_size, stride=2,
                              kernel_size=3,
                              padding=1)
        self.P6 = nn.Conv2d(in_channels=c5_ch,
                            out_channels=self.feat_size,
                            kernel_size=3, stride=2, padding=pad_out(3))
        self.P5_1 = nn.Conv2d(in_channels=c5_ch,
                              out_channels=self.feat_size,
                              kernel_size=1, padding=pad_out(1))

        self.P5_2 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size,
                              kernel_size=3, padding=pad_out(3))

        self.P4_1 = nn.Conv2d(in_channels=c4_ch,
                              out_channels=self.feat_size, kernel_size=1,
                              padding=pad_out(1))

        self.P4_2 = nn.Conv2d(in_channels=self.feat_size,
                              out_channels=self.feat_size, kernel_size=3,
                              padding=pad_out(3))

        self.P3_1 = nn.Conv2d(in_channels=c3_ch,
                              out_channels=self.feat_size, kernel_size=1,
                              padding=pad_out(1))

        self.P3_2 = nn.Conv2d(in_channels=self.feat_size,
                              out_channels=self.feat_size, kernel_size=3,
                              padding=pad_out(3))

    def forward(self, c3, c4, c5):
        p51 = self.P5_1(c5)
        p5_out = self.P5_2(p51)

        # p5_up = F.interpolate(p51, scale_factor=2)
        p5_up = F.interpolate(p51, size=(c4.size(2), c4.size(3)))
        p41 = self.P4_1(c4) + p5_up
        p4_out = self.P4_2(p41)

        # p4_up = F.interpolate(p41, scale_factor=2)
        p4_up = F.interpolate(p41, size=(c3.size(2), c3.size(3)))
        p31 = self.P3_1(c3) + p4_up
        p3_out = self.P3_2(p31)

        p6_out = self.P6(c5)

        p7_out = self.P7_2(F.relu(p6_out))

        # p8_out = self.p8_gen(F.relu(p7_out))
        p8_out = F.adaptive_avg_pool2d(p7_out, 1)
        return [p3_out, p4_out, p5_out, p6_out, p7_out, p8_out]






class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.Linear1 = nn.Linear(self.state_size, 2048)
        self.Linear2 = nn.Linear(2048, 512)
        self.Linear3 = nn.Linear(512, self.action_size)

        #self.resnet = models.resnet50(pretrained=True).cuda().eval()
        self.resnet = models.resnet50(pretrained=True).cuda()

        self.fpn = FPN_backbone(512, 1024, 2048)


    def extract_visual_feat(self, inputs):
        output = self.resnet.conv1(inputs)
        output = self.resnet.bn1(output)
        output = self.resnet.relu(output)
        output = self.resnet.maxpool(output)
        output_1 = self.resnet.layer1(output)
        output_2 = self.resnet.layer2(output_1)
        output_3 = self.resnet.layer3(output_2)
        output_4 = self.resnet.layer4(output_3)
        output_avg = self.resnet.avgpool(output_4)
        #output_avg = output_avg.reshape(-1, 2048).detach()
        output_avg = output_avg.reshape(-1, 2048)
        return output_1, output_2, output_3, output_4, output_avg

    def forward(self, ref_imgs_tensor, sent_feats, location_tensor, history_actions_tensor):

        output_1, output_2, output_3, output_4, output_avg = self.extract_visual_feat(ref_imgs_tensor)

        # for 2048
        state_feats = torch.cat((output_avg, sent_feats, location_tensor), 1)
        output = F.relu(self.Linear1(state_feats))
        output = F.relu(self.Linear2(output))
        actions_tensor = self.Linear3(output)




        '''
        # for fpn
        feat = self.fpn(output_2, output_3, output_4)

        layer2_fused_feat = F.adaptive_avg_pool2d(feat[0], 1).reshape(-1, 256)
        layer2_state_feats = torch.cat((layer2_fused_feat, sent_feats, location_tensor), 1)
        layer2_output = F.relu(self.Linear1(layer2_state_feats))
        layer2_output = F.relu(self.Linear2(layer2_output))
        layer2_actions_tensor = self.Linear3(layer2_output)

        #actions_tensor = layer2_actions_tensor

        layer3_fused_feat = F.adaptive_avg_pool2d(feat[1], 1).reshape(-1, 256)
        layer3_state_feats = torch.cat((layer3_fused_feat, sent_feats, location_tensor), 1)
        layer3_output = F.relu(self.Linear1(layer3_state_feats))
        layer3_output = F.relu(self.Linear2(layer3_output))
        layer3_actions_tensor = self.Linear3(layer3_output)
        
        layer4_fused_feat = F.adaptive_avg_pool2d(feat[2], 1).reshape(-1, 256)
        layer4_state_feats = torch.cat((layer4_fused_feat, sent_feats, location_tensor), 1)
        layer4_output = F.relu(self.Linear1(layer4_state_feats))
        layer4_output = F.relu(self.Linear2(layer4_output))
        layer4_actions_tensor = self.Linear3(layer4_output)

        actions_tensor = layer2_actions_tensor + layer3_actions_tensor + layer4_actions_tensor
        '''



        actions_cat = Categorical(F.softmax(actions_tensor, dim=-1))

        return actions_tensor, actions_cat



class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()

        self.state_size = state_size

        self.Linear1 = nn.Linear(self.state_size, 2048)
        self.Linear2 = nn.Linear(2048, 512)
        self.Linear3 = nn.Linear(512, 1)

        #self.resnet = models.resnet50(pretrained=True).cuda().eval()
        self.resnet = models.resnet50(pretrained=True).cuda()

        self.fpn = FPN_backbone(512, 1024, 2048)


    def extract_visual_feat(self, inputs):
        output = self.resnet.conv1(inputs)
        output = self.resnet.bn1(output)
        output = self.resnet.relu(output)
        output = self.resnet.maxpool(output)
        output_1 = self.resnet.layer1(output)
        output_2 = self.resnet.layer2(output_1)
        output_3 = self.resnet.layer3(output_2)
        output_4 = self.resnet.layer4(output_3)
        output_avg = self.resnet.avgpool(output_4)
        #output_avg = output_avg.reshape(-1, 2048).detach()
        output_avg = output_avg.reshape(-1, 2048)
        return output_1, output_2, output_3, output_4, output_avg

    def forward(self, ref_imgs_tensor, sent_feats, location_tensor, history_actions_tensor):

        output_1, output_2, output_3, output_4, output_avg = self.extract_visual_feat(ref_imgs_tensor)


        # for 2048
        state_feats = torch.cat((output_avg, sent_feats, location_tensor), 1)
        output = F.relu(self.Linear1(state_feats))
        output = F.relu(self.Linear2(output))
        values_tensor = self.Linear3(output)


        '''
        # for fpn
        feat = self.fpn(output_2, output_3, output_4)

        layer2_fused_feat = F.adaptive_avg_pool2d(feat[0], 1).reshape(-1, 256)
        layer2_state_feats = torch.cat((layer2_fused_feat, sent_feats, location_tensor), 1)
        layer2_output = F.relu(self.Linear1(layer2_state_feats))
        layer2_output = F.relu(self.Linear2(layer2_output))
        layer2_values_tensor = self.Linear3(layer2_output)

        #values_tensor = layer2_values_tensor

        layer3_fused_feat = F.adaptive_avg_pool2d(feat[1], 1).reshape(-1, 256)
        layer3_state_feats = torch.cat((layer3_fused_feat, sent_feats, location_tensor), 1)
        layer3_output = F.relu(self.Linear1(layer3_state_feats))
        layer3_output = F.relu(self.Linear2(layer3_output))
        layer3_values_tensor = self.Linear3(layer3_output)

        layer4_fused_feat = F.adaptive_avg_pool2d(feat[2], 1).reshape(-1, 256)
        layer4_state_feats = torch.cat((layer4_fused_feat, sent_feats, location_tensor), 1)
        layer4_output = F.relu(self.Linear1(layer4_state_feats))
        layer4_output = F.relu(self.Linear2(layer4_output))
        layer4_values_tensor = self.Linear3(layer4_output)

        values_tensor = layer2_values_tensor + layer3_values_tensor + layer4_values_tensor
        '''

        return values_tensor


class Refine(nn.Module):
    def __init__(self, state_size):
        super(Refine, self).__init__()

        self.state_size = state_size

        self.Linear1 = nn.Linear(self.state_size, 2048)
        self.Linear2 = nn.Linear(2048, 512)
        self.Linear3 = nn.Linear(512, 4)

        # self.resnet = models.resnet50(pretrained=True).cuda().eval()
        self.resnet = models.resnet50(pretrained=True).cuda()

        self.fpn = FPN_backbone(512, 1024, 2048)

    def extract_visual_feat(self, inputs):
        output = self.resnet.conv1(inputs)
        output = self.resnet.bn1(output)
        output = self.resnet.relu(output)
        output = self.resnet.maxpool(output)
        output_1 = self.resnet.layer1(output)
        output_2 = self.resnet.layer2(output_1)
        output_3 = self.resnet.layer3(output_2)
        output_4 = self.resnet.layer4(output_3)
        output_avg = self.resnet.avgpool(output_4)
        # output_avg = output_avg.reshape(-1, 2048).detach()
        output_avg = output_avg.reshape(-1, 2048)
        return output_1, output_2, output_3, output_4, output_avg

    def forward(self, ref_imgs_tensor, sent_feats, location_tensor):
        output_1, output_2, output_3, output_4, output_avg = self.extract_visual_feat(ref_imgs_tensor)


        # for 2048
        state_feats = torch.cat((output_avg, sent_feats, location_tensor), 1)
        output = F.relu(self.Linear1(state_feats))
        output = F.relu(self.Linear2(output))
        coordinate_tensor = self.Linear3(output)


        '''
        # for fpn
        feat = self.fpn(output_2, output_3, output_4)

        layer2_fused_feat = F.adaptive_avg_pool2d(feat[0], 1).reshape(-1, 256)
        layer2_state_feats = torch.cat((layer2_fused_feat, sent_feats, location_tensor), 1)
        layer2_output = F.relu(self.Linear1(layer2_state_feats))
        layer2_output = F.relu(self.Linear2(layer2_output))
        layer2_actions_tensor = self.Linear3(layer2_output)

        #coordinate_tensor = layer2_actions_tensor

        layer3_fused_feat = F.adaptive_avg_pool2d(feat[1], 1).reshape(-1, 256)
        layer3_state_feats = torch.cat((layer3_fused_feat, sent_feats, location_tensor), 1)
        layer3_output = F.relu(self.Linear1(layer3_state_feats))
        layer3_output = F.relu(self.Linear2(layer3_output))
        layer3_actions_tensor = self.Linear3(layer3_output)

        layer4_fused_feat = F.adaptive_avg_pool2d(feat[2], 1).reshape(-1, 256)
        layer4_state_feats = torch.cat((layer4_fused_feat, sent_feats, location_tensor), 1)
        layer4_output = F.relu(self.Linear1(layer4_state_feats))
        layer4_output = F.relu(self.Linear2(layer4_output))
        layer4_actions_tensor = self.Linear3(layer4_output)

        coordinate_tensor = layer2_actions_tensor + layer3_actions_tensor + layer4_actions_tensor
        '''

        return coordinate_tensor


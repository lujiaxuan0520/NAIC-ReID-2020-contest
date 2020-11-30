from __future__ import absolute_import
from __future__ import division
import math
from torch.nn.parameter import Parameter
from HGNN import HGNN_conv
from HGNN import construct_H_with_KNN, generate_G_from_H

__all__ = ['vmgn_hgnn']

import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo

from torchreid.utils.reidtools import calc_splits
from torchreid.utils.torchtools import weights_init_kaiming, weights_init_classifier

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    """
    Residual network
    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, block, layers, last_stride=2):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class GSTA_HGNN(nn.Module):
    def __init__(self, arch, num_classes, block, layers,
                 num_split, pyramid_part, use_pose, learn_graph,
                 consistent_loss, isFinal=False,global_branch=False,
                 m_prob=1.0, K_neigs=[10], is_probH=True,
                 dropout=0.5,learn_attention=True,**kwargs):
        self.inplanes = 64
        super(GSTA_HGNN, self).__init__()
        self.loss = {'xent', 'htri'}
        self.feature_dim = 512 * block.expansion
        self.use_pose = use_pose
        self.learn_graph = learn_graph
        self.learn_attention = learn_attention
        self.training = False
        self.learn_edge = False
        self.isFinal = isFinal
        self.global_branch = global_branch

        # backbone network
        backbone = ResNetBackbone(block, layers, 1)
        init_pretrained_weights(backbone, model_urls[arch])
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4_1 = backbone.layer4
        self.layer4_2 = copy.deepcopy(self.layer4_1)

        # global branch, from layer4_1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.global_bottleneck.bias.requires_grad_(False)
        self.global_classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
        weights_init_kaiming(self.global_bottleneck)
        weights_init_classifier(self.global_classifier)

        # split branch, from layer4_2
        self.num_split = num_split
        self.total_split_list = calc_splits(num_split) if pyramid_part else [num_split]
        self.total_split = sum(self.total_split_list)

        self.parts_avgpool = nn.ModuleList()
        for n in self.total_split_list:
            self.parts_avgpool.append(nn.AdaptiveAvgPool2d((n, 1)))

        # hgnn graph layers
        self.m_prob = m_prob  # parameter in hypergraph incidence matrix construction
        self.K_neigs = K_neigs  # the number of neighbor expansion
        self.is_probH = is_probH  # probability Vertex-Edge matrix or binary
        self.dropout = dropout
        self.hgc1 = HGNN_conv(self.feature_dim, self.feature_dim)  # self.hgc1 = HGNN_conv(self.feature_dim, self.n_hid)
        self.hgc2 = HGNN_conv(self.feature_dim, self.feature_dim)  # self.hgc2 = HGNN_conv(self.n_hid, num_classes)

        # attention branch
        if self.learn_attention:
            self.attention_weight = Parameter(torch.Tensor(self.seq_len, self.total_split,1))
            self._reset_attention_parameters()

        self.consistent_loss = consistent_loss

        self.att_bottleneck = nn.BatchNorm1d(self.feature_dim) # nn.BatchNorm1d(num_classes)
        self.att_bottleneck.bias.requires_grad_(False)
        self.att_classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
        weights_init_kaiming(self.att_bottleneck)
        weights_init_classifier(self.att_classifier)

    def _attention_op(self, feat):
        """
        do attention fusion
        :param feat: (batch, seq_len, num_split, c)
        :return: feat: (batch, num_split, c)
        """
        att = F.normalize(feat.norm(p=2, dim=2, keepdim=True), p=1, dim=1)
        f = feat.mul(att).sum(dim=1)
        return f

    def _learn_attention_op(self,feat):
        """
        do attention fusion, with the weight learned
        :param feat: (batch, seq_len, num_split, c)
        :return: feat: (batch, num_split, c)
        """
        f = feat.mul(self.attention_weight)
        f = f.sum(dim=1)
        return f

    def _reset_attention_parameters(self):
        stdv = 1. / math.sqrt(self.attention_weight.size(1))
        self.attention_weight.data.uniform_(-stdv, stdv)
        # nn.init.normal_(self.attention_weight.data, 0.1,0.001)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x4_1 = self.layer4_1(x)
        x4_2 = self.layer4_2(x)
        return x4_1, x4_2

    def forward(self, x, *args):
        B, C, H, W = x.size()
        x = x.view(B, C, H, W) # change the size of tensor
        x4_1, x4_2 = self.featuremaps(x) # features of the input sequence
        _, c, h, w = x4_1.shape

        # global branch
        if self.global_branch:
            x4_1 = x4_1.view(B, c, h, w).contiguous()
            g_f = self.global_avg_pool(x4_1).view(B, -1)
            g_bn = self.global_bottleneck(g_f)

        # split branch
        v_f = list()
        for idx, n in enumerate(self.total_split_list):
            v_f.append(self.parts_avgpool[idx](x4_2).view(B, c, n))
        v_f = torch.cat(v_f, dim=2)
        f = v_f.transpose(1, 2).contiguous()

        # construct the hgnn graph
        G = []
        for feature in f:
            H = construct_H_with_KNN(feature.cpu().detach().numpy(), K_neigs=self.K_neigs, is_probH=self.is_probH,
                                     m_prob=self.m_prob)
            g = generate_G_from_H(H,variable_weight=self.learn_edge)
            # G.append(g.A)
            G.append(g)
        G = torch.stack(G)
        # G = torch.tensor(G, dtype=torch.float32)

        # if self.use_pose:
        #     G = F.normalize(G, p=1, dim=2)
        #     adj = F.normalize(adj, p=1, dim=2)
        #     if adj.is_cuda:
        #         G = G.to('cuda')
        #     G = (adj + G) / 2

        # hgnn graph propogation
        f = F.relu(self.hgc1(f, G))
        f = F.dropout(f, self.dropout)
        f = self.hgc2(f, G)

        # f = f.view(B, S, self.total_split, f.shape[-1])

        # attention branch
        if self.learn_attention: # learn the weight of attention
            f_fuse = self._learn_attention_op(f)
        else: # calculate the norm as the attention weight
            f_fuse = self._attention_op(f)

        att_f = f_fuse.view(B, -1)
        att_bn = self.att_bottleneck(att_f)

        if not self.training  or self.isFinal:
            if self.global_branch: # use two branch
                return torch.cat([g_bn, att_bn], dim=1)
            else:
                return att_bn

        if self.global_branch:
            g_out = self.global_classifier(g_bn)
        att_out = self.att_classifier(att_bn)

        # # consistent
        # if self.consistent_loss and self.training:
        #     satt_f_list = list()
        #     satt_out_list = list()
        #     # random select sub frames
        #     assert S >= 5
        #     for num_frame in [S-3, S-2, S-1]:
        #         sub_index = torch.randperm(S)[:num_frame]
        #         sub_index = torch.sort(sub_index)[0]
        #         sub_index = sub_index.long().to(f.device)
        #         sf = torch.gather(f, dim=1, index=sub_index.view(1, num_frame, 1, 1).repeat(B, 1, self.total_split, c))
        #         # sf_fuse = self._learn_attention_op(sf) if self.learn_attention else self._attention_op(sf)
        #         sf_fuse = self._attention_op(sf)
        #         satt_f = sf_fuse.mean(dim=1).view(B, -1)
        #         satt_bn = self.att_bottleneck(satt_f)
        #         satt_out = self.att_classifier(satt_bn)
        #         satt_f_list.append(satt_f)
        #         satt_out_list.append(satt_out)

        if self.loss == {'xent'}:
            out_list = [g_out, att_out] if self.global_branch else [att_out]
            # if self.consistent_loss:
            #     out_list.extend(satt_out_list)
            return out_list
        elif self.loss == {'xent', 'htri'}:
            out_list = [g_out, att_out] if self.global_branch else [att_out]
            f_list = [g_f, att_f] if self.global_branch else [att_f]
            # if self.consistent_loss:
            #     out_list.extend(satt_out_list)
            #     f_list.extend(satt_f_list)
            return out_list, f_list
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_url))


def vmgn_hgnn(num_classes, isFinal=False, global_branch=False, arch="resnet50"):
    # arch = "resnet50" # can be changed
    layers_dict = {"resnet50":[3, 4, 6, 3],
                   "resnet101":[3, 4, 23, 3],
                   "resnet152":[3, 8, 36, 3]}
    layers = layers_dict[arch]
    model = GSTA_HGNN(
        arch = arch,
        num_classes=num_classes,
        block=Bottleneck,
        layers=layers,
        last_stride=1,
        num_split=8,
        pyramid_part=True,
        num_gb=2,
        use_pose=False,
        learn_graph=True,
        consistent_loss=False,
        m_prob=1.0,
        K_neigs=[3],
        is_probH=True,
        dropout=0.5,
        learn_attention=False,
        isFinal = isFinal,
        global_branch=global_branch
    )

    return model
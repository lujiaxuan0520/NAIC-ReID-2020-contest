from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


def adjust_learning_rate(optimizer, base_lr, epoch, stepsize, gamma=0.1):
    # decay learning rate by 'gamma' for every 'stepsize'
    lr = base_lr * (gamma ** (epoch // stepsize))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def count_num_param(model):
    num_param = sum(p.numel() for p in model.parameters()) / 1e+06
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
    return num_param

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.001)
            nn.init.constant_(m.bias, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.001)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
# coding:utf8
import torch
from torch.nn import init
import torch as t
from torch import nn
import time
import torch.nn as nn
from torch.nn.functional import mse_loss

def l2_loss(outputs, target, mask):
    '''
    @param outputs: tuple of variabel size(6),outputs of 6 stages
    @param target: tensor  size(h,w,)
    @param mask: Variable
    @return loss
    '''    

    losses = []
    batch_size,c_,h_,w_ = outputs[0].size()

    for output in outputs:
        dist = (output - target) * mask#.view(-1,1,h_,w_).expand_as(output)
        losses.append((dist**2).sum()/dist.numel())
    loss = sum(losses)
    return loss,losses

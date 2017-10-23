from itertools import chain
import torch
import torchvision as tv
import numpy as np
from .yellowfin import YFOptimizer


def get_optimizer(model, lr, lr1=5e-5, weight_decay=1e-5):
    parameters = model.parameters()
    optimizer = torch.optim.Adam([
                {'params': model.model0.model.parameters(), 'lr' : lr1},
                {'params': model.model1.pre_model.parameters(), 'lr': lr1},
                {'params': model.model1.cmp_model.parameters()},
                {'params': model.model1.paf_model.parameters()},
                
                {'params': model.model2.pre_model.parameters(), 'lr': lr1},
                {'params': model.model2.cmp_model.parameters()},
                {'params': model.model2.paf_model.parameters()},
                
                {'params': model.model3.pre_model.parameters(), 'lr': lr1},
                {'params': model.model3.cmp_model.parameters()},
                {'params': model.model3.paf_model.parameters()},
                
                {'params': model.model4.pre_model.parameters(), 'lr': lr1},
                {'params': model.model4.cmp_model.parameters()},
                {'params': model.model4.paf_model.parameters()},
                
                {'params': model.model5.pre_model.parameters(), 'lr': lr1},
                {'params': model.model5.cmp_model.parameters()},
                {'params': model.model5.paf_model.parameters()},
                
                {'params': model.model6.pre_model.parameters(), 'lr': lr1},
                {'params': model.model6.cmp_model.parameters()},
                {'params': model.model6.paf_model.parameters()},
            ], lr = lr, weight_decay = weight_decay, betas=(0.9, 0.99))
    # optimizer = torch.optim.Adam(
    #     parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    return optimizer


def get_yellow(model, lr):
    return YFOptimizer(model.parameters(), weight_decay=1e-4)

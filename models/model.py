#coding:utf8

import torchvision as tv
import torch as t
from torch import nn
from mobilenet import MobileNet
from collections import namedtuple
from basic_module import BasicModule

class Stage(nn.Module):
    
    def __init__(self, inplanes,outplanes=41):
        super(Stage, self).__init__()
        self.cf = nn.Sequential(
            nn.Conv2d(inplanes,128,7,1,3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128,128,7,1,3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128,128,7,1,3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128,128,7,1,3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128,128,7,1,3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128,outplanes,1)
        )


    def forward(self, x):
        return self.cf(x)


class KeypointModel(BasicModule):
    def __init__(self,opt):
        super(KeypointModel, self).__init__(opt)
        self.pretrained= MobileNet()
        self.trf = nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        # self.ReturnType = namedtuple('ReturnType',['out1','out2','out3','out4','out5','out6'])
        stages = [Stage(128)] +  [Stage(169) for _ in range(2,7)]
        self.stages = nn.ModuleList(stages)

    def forward(self,img):
        img = self.pretrained(img)
        #if self.optimizer.param_groups[0]['lr'] == 0:
        #    img = img.detach()
        features = self.trf(img)

        output = self.stages[0](features)
        outputs= [output]
        for ii in range(1,6):
            stage = self.stages[ii]
            input = t.cat([features,output],dim=1)
            output = stage(input)
            outputs.append(output)
        
        return outputs

    def get_optimizer(self,lr1,lr2):
        param_groups = [
            {'params':self.pretrained.parameters(),'lr':lr1},
            {'params':self.stages.parameters(),'lr':lr2},
            {'params':self.trf.parameters(),'lr':lr2}
        ]

        self.optimizer = t.optim.Adam(param_groups)
        return self.optimizer

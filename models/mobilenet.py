#coding:utf8

import torchvision as tv
import torch as t
from torch import nn


class MobileNet(nn.Module):

    def __init__(self):
        super(MobileNet, self).__init__()
        self.model_name='mobilenet'

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.classifier = nn.Linear(1024, 2)
        self.load_state_dict(t.load('/data_ssd/ai_challenger/ai_challenger_keypoint_train_20170909/mobilenet.pth'))
        self.pack()

    def pack(self):
        del self.classifier
        for ii in range(13,5,-1):
            del self.features._modules[str(ii)]
        #del self.features._modules[14]


    def forward(self, x):
        x = self.features(x)
        # x = self.classifier(x)
        return x

# encoding:utf-8
import torch as t
import torch
from torch.utils import data
import numpy as np
from skimage import exposure
from PIL import Image, ImageEnhance
import random
from config import opt
from fastitem import FastItem
from torchvision import transforms
from skimage import transform as sktrf

class Dataset(data.Dataset):
    def __init__(self, opt):
        self.annotations = t.load(opt.anno_path)
        self.opt = opt
        self.transforms = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
                 ])

    def __getitem__(self, index):
        """
            :return img: 3xHxW
            :return gt: 41xHxW
            :return weight: HxW
        """
        anno = self.annotations[index]
        item = FastItem(self.opt, anno)
        img, cf, paf, weight = item.get()

        # scale
#        paf = item.rescale(paf,0.125)
 #       cf = item.rescale(cf,0.125)
  #      weight = item.rescale(weight,0.125)
        
        # to tensor
        img = t.from_numpy(img.transpose((2, 0, 1)))
        cf = t.from_numpy(cf.transpose((2, 0, 1)))
        paf = t.from_numpy(paf.transpose((2, 0, 1)))
        weight = t.from_numpy(weight).squeeze()

        # other process
        img = self.transforms(img)
        gt = t.cat([cf, paf], dim=0)
        
        return img.float(), gt.float(), weight.unsqueeze(0).float().expand_as(gt)

    def __len__(self):
        return len(self.annotations)

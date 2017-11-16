#coding:utf8
import math
import sys
import time
from config import opt as opt_
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch as t

import torchvision
from data.tool import show_paf
from models import KeypointModel
from utils.test_tool import *
import tqdm
import json

class TestConfig:
    scale_search = [0.8,1,1.5,2.0]
    boxsize=384
    stride=8
    val = True
    val_dir = '/data/image/ai_cha/old_del/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/'
    ckpt_path = 'checkpoints/KeypointModel_1030_0332_0.0100910376112.pth'
    output_path = 'result/'
    test_num = 1000


opt = TestConfig()


def test_file(pose_model,file_name='206998b0cca06ec6f3951e4acf7e178e114bdba7.jpg'):
    """
    @param file_name: 206998b0cca06ec6f3951e4acf7e178e114bdba7.jpg
    @param file_name: 
    """

    # 图片
    oriImg = cv.imread(file_name) # B,G,R order
    multiplier = [x * opt.boxsize*1.0 / oriImg.shape[0] for x in opt.scale_search]

    # 得到heatmap+paf
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 26))
    for scale in multiplier:
        (heatmap1,heatmap2),(paf1,paf2) = get_output(pose_model,scale,oriImg,opt.stride)
        heatmap = flip_heatmap(heatmap1,heatmap2)
        paf = flip_paf(paf1,paf2)
        heatmap_avg += heatmap
        paf_avg += paf
    heatmap_avg /= len(multiplier)
    paf_avg /= len(multiplier)
    result = dict(heatmap=heatmap_avg,paf=paf_avg)
    t.save(result,'%s/%s.pth' %(opt.output_path,file_nam))
    # # 找到关键点
    # all_peaks,peak_counter = find_peaks(heatmap_avg,0.1)
    # # 找到连接
    # special_k,connection_all = find_connection(all_peaks,paf_avg)
    # # 找到人
    # subsets,candidate = find_person(all_peaks,special_k,connection_all)
    # return get_result(subsets,candidate,file_name.split('/')[-1][:-4])

def main(**kwargs):
    pose_model=KeypointModel(opt_)
    pose_model.load(opt.ckpt_path).eval().cuda()
    
    img_list = glob.glob(opt.val_dir + '*.jpg')[:opt.test_num]
    
    for ii,img in tqdm.tqdm(enumerate(img_list)):
        test_file(pose_model, img)

if __name__=='__main__':
    import fire
    fire.Fire()

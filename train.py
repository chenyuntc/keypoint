# coding:utf8
import os
import time
import fire
import ipdb

import numpy as np
import torch as t
import torchnet as tnt
from tqdm import tqdm
from skimage import transform
import matplotlib
matplotlib.use('agg')

from config import opt
from data import Dataset
from models import l2_loss
from utils import Visualizer,mask_img
import models
from data import tool




def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    model = models.KeypointModel(opt)
    if opt.model_path is not None:
        model.load(opt.model_path)

    model.cuda()
    dataset = Dataset(opt)
    dataloader = t.utils.data.DataLoader(
        dataset,
        opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
        drop_last=True)

    lr1, lr2 = opt.lr1, opt.lr2
    optimizer = model.get_optimizer(lr1, lr2)
    loss_meter = tnt.meter.AverageValueMeter()
    pre_loss = 1e100
    model.save()
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        start = time.time()

        for ii, (img, gt, weight) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            img = t.autograd.Variable(img).cuda()
            target = t.autograd.Variable(gt).cuda()
            weight = t.autograd.Variable(weight).cuda()
            outputs = model(img)
            loss, loss_list = l2_loss(outputs, target, weight)
            #(loss).backward()
            loss_meter.add(loss.data[0])
            #optimizer.step()

            # 可视化， 记录， log，print
            if ii % opt.plot_every == 0 and ii > 0:
                if os.path.exists(opt.seg_debug_file):
                    ipdb.set_trace()
                vis_plots = {'loss': loss_meter.value()[0], 'ii': ii}
                vis.plot_many(vis_plots)

                # 随机展示一张图片
                k = t.randperm(img.size(0))[0]
                show = img.data[k].cpu()
                raw = (show*0.225+0.45).clamp(min=0,max=1)
            
                train_masked_img =  mask_img( raw,outputs[-1].data[k][14])
                origin_masked_img = mask_img( raw,gt[k][14])

                vis.img('target',origin_masked_img)
                vis.img('train',train_masked_img)
                vis.img('label',gt[k][14])
                vis.img('predict',outputs[-1].data[k][14].clamp(max=1,min=0))
                paf_img = tool.vis_paf(raw,gt[k][15:])
                train_paf_img = tool.vis_paf(raw,outputs[-1][k].data[15:].clamp(min=-1,max=1))
                vis.img('paf_train',train_paf_img)
                #fig = tool.show_paf(np.transpose(raw.cpu().numpy(),(1,2,0)),gt[k][15:].cpu().numpy().transpose((1,2,0))).get_figure()
                #paf_img = tool.fig2data(fig).astype(np.int32)
                #vis.img('paf',t.from_numpy(paf_img/255).float())
                vis.img('paf',paf_img)
        model.save(loss_meter.value()[0])
        vis.save([opt.env])

if __name__ == "__main__":
    fire.Fire()

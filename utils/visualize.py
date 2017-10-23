#coding:utf8
from itertools import chain
import visdom
import torch
import time
import torchvision as tv
import numpy as np
from PIL import Image
from skimage import transform
to_pil = tv.transforms.ToPILImage()
to_ten = tv.transforms.ToTensor()

def mask_img(img,mask,scale=8):
    img = to_pil(img)
    mask = np.abs(mask.cpu().numpy())

    mask = mask/mask.max()
    alpha = transform.rescale((mask>0.5)*1.,8,mode='reflect')*192
    alpha = np.uint8(alpha)
    alpha = Image.fromarray(alpha)
    
    mask_ = np.uint8(transform.rescale(mask,8,mode='reflect')*255)  
    mask_ =  Image.fromarray(mask_)
    masked_img = Image.composite(mask_,img,alpha)

    return to_ten(masked_img)

class Visualizer():
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    '''

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, **kwargs)
        
        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {} 
        self.log_text = ''
    def reinit(self,env='default',**kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env,**kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        '''
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y):
        '''
        self.plot('loss',1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=unicode(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def img(self, name, img_):
        '''
        self.img('input_img',t.Tensor(64,64))
        '''
         
        if len(img_.size())<3:
            img_ = img_.cpu().unsqueeze(0) 
        self.vis.image(img_.cpu(),
                       win=unicode(name),
                       opts=dict(title=name)
                       )
    def img_grid_many(self,d):
        for k, v in d.iteritems():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        '''
        一个batch的图片转成一个网格图，i.e. input（36，64，64）
        会变成 6*6 的网格图，每个格子大小64*64
        '''
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1,min=0)))

    def log(self,info,win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m%d_%H%M%S'),\
                            info=info)) 
        self.vis.text(self.log_text,win='log_text')   

    def __getattr__(self, name):
        return getattr(self.vis, name)


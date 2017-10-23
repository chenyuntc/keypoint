# encoding:utf-8
import time
tfmt = '%m%d_%H%M%D'


class Config:
    crop_size_x = 384
    crop_size_y = 384

    sigma = 1.0
    boxsize = 368
    env = 'keypoint'  # Visdom env
    model = 'KeypointModel'
    model_path = None
    batch_size = 32
    num_workers = 10
    shuffle = True
    seg_debug_file = '/tmp/debugbk'
    lr1 = 0
    lr2 = 1e-3
    max_epoch = 100
    model_path = None
    
    plot_every = 10
    save_every = 3000
    decay_every = 1500

    img_root = '/data_ssd/ai_challenger/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902/'
#    img_root = '/mnt/7/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902/'
    anno_path = '/data_ssd/ai_challenger/ai_challenger_keypoint_train_20170909/train.pth'

    downsample_rate = 8

def parse(self,kwargs):
    # 处理配置和参数
    for k, v in kwargs.iteritems():
        if not hasattr(self, k):
            print("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)
    
    keys = sorted(self.state_dict().keys())
    for key in keys:
        if not key.startswith('__'):
            print(key, getattr(self, key))

def state_dict(self):
    return  {k:getattr(self,k) for k in dir(self) if not k.startswith('_') and k!='parse' and k!='state_dict' }

Config.parse  = parse
Config.state_dict = state_dict
opt = Config()

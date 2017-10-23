#coding:utf8
import torch as t
import time
class BasicModule(t.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self,opt=None):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self).__name__)# 默认名字
        self.opt = opt
        
    def load(self, path,map_location=lambda storage, loc: storage):
        checkpoint = t.load(path,map_location=map_location)
        if 'opt' in checkpoint:
            self.load_state_dict(checkpoint['d'])
            print('old config：')
            print(checkpoint['opt'])
        else:
            self.load_state_dict(checkpoint)
        # for k,v in checkpoint['opt'].items():
        #     setattr(self.opt,k,v)
        return self

    def save(self, name=''):
        # prefix = 'checkpoint/'+name
        file_name = 'checkpoints/{model_name}_{time_}_{name}.pth'.format(
                                        time_=time.strftime('%m%d_%H%M'), 
                                        model_name = self.model_name,
                                        name= str(name))
        state_dict = self.state_dict()
        opt_state_dict = self.opt.state_dict()
        t.save({'d':state_dict,'opt':opt_state_dict}, file_name)
        return file_name

#coding=utf-8
import numpy as np
from PIL import Image
import copy
from scipy import stats
from skimage import draw,transform,util


class FastItem(object):
    '''
    每一张图片对应一个Item对象
    '''
    def __init__(self,opt,anno):
        '''
        @param: opt
        @param: anno 标记 dict对象
        '''
        self.anno = copy.deepcopy(anno)# 不要修改之前的dict对象
        self._img = Image.open(opt.img_root+anno['image_id']+'.jpg')
        
        
        self.size = list(self._img.size)[::-1] # 图片形状 height*weight
        self.output_size = opt.crop_size_x,opt.crop_size_y # 图片输出形状
        self.downsample_rate = opt.downsample_rate
        self.opt = opt



        size,output_size = np.array(self.size),np.array(self.output_size)
        ratio = (size+0.0)/output_size
        ratio = min(ratio)
        re_size = (size/ratio)
        self.re_size =(re_size/opt.downsample_rate).astype(np.int32)
        self.update_anno(opt.downsample_rate*ratio)
                
        # !TODO 针对不同的躯干设计不同的比例
        # 手臂的可以比例高点,肩膀到左髋的距离比较大，粗细可以小一点
        self.ratio = 0.1 # 长/宽为3 即手臂长/手臂粗

    def update_anno(self,ratio):
        '''
        调整anno信息:
        之前的信息座标都是先列,再行,现在统一改成先行再列
        形如:  
        {u'image_id': u'043758c591b58f39a01648c49b5154ad1e01d400', u'keypoint_annotations': 
        {u'human1': [(x1, y1, type1), ....(x14, y14, type14)]}
        , u'human_annotations': {u'human1': [x, y, x, y]}}     
        '''

        for human,human_anno in self.anno['human_annotations'].items():
            human_anno[1],human_anno[0], human_anno[3],human_anno[2] =  \
                            [int(_/ratio) for _ in human_anno]            

        key_annos = self.anno['keypoint_annotations']
        for human,key_anno in key_annos.items():
            coord2 = [int(_/ratio) for _ in key_anno[::3]]
            coord1 = [int(_/ratio) for _ in  key_anno[1::3]]
            stat_ = key_anno[2::3]
            key_annos[human] = zip(coord1,coord2,stat_)
    
    def confidence_map(self):
        '''
        针对每个关节,生成类似高斯的激活
        !TODO: 输出的尺度要求? 现在太小,是否要增大
        '''
        _h,_w = self.re_size
        _h,_w = int(_h),int(_w)
        # _h,_w = _h/self.opt.downsample_rate,_w/self.opt.downsample_rate 
        part_maps = np.zeros([_h,_w,15])
        for part in range(14):
            keypoint_anno = self.anno['keypoint_annotations']
            person_num = len(keypoint_anno)
            maps = np.zeros([_h,_w,person_num])
            for ii,(human,middle) in enumerate(keypoint_anno.items()):
                if middle[part][2]==3:
                    # 不在图中的初始化为零
                    map = np.zeros([_h,_w])
                else:
                    middle_ = middle[part][:2]
                    map = self._gaussion_map((_h,_w),middle_,self.opt.sigma)
                maps[:,:,ii] = map
            part_map = maps.max(axis=2)
            part_maps[:,:,part]= part_map
        
        part_maps[:,:,-1] = part_maps[:,:,:14].max(axis=2)
        self.cf = part_maps
        return self.cf
        
    def _gaussion_map(self,shape,middle,sigma):
        '''
        根据(u,sigma)和形状生成高斯热度图

        @sigma: 标准差！ 不是方差
        @shape: 生成map的形状
        @middle：中心位置（均值）
        '''
        middle = np.array(middle).reshape([1,1,-1])
        def gaussion(x,y,middle=middle):
            x = x[...,np.newaxis]
            y = y[...,np.newaxis]
            coord = np.concatenate([x,y],axis=2)
            distance = np.linalg.norm(coord - middle, axis=2)
            #return stats.norm.pdf(distance, 0, sigma)*(2*np.pi*(sigma**0.5))
            return stats.norm.pdf(distance, 0, sigma)*((2*np.pi)**0.5)*sigma
        return np.fromfunction(gaussion,(shape))
    
    def PAF_map(self):
        '''
        生成paf,
        '''
        mid_1 = [12, 0, 3, 0, 1, 3, 4, 0, 6, 7, 3, 9, 10]
        mid_2 = [13, 13, 13, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
        h_,w_ = self.re_size
        # h_,w_ = h_/self.opt.downsample_rate/self.ratio,w_/self.opt.downsample_rate/self.ratio

        keypoint_anno = self.anno['keypoint_annotations']
        person_num = len(keypoint_anno)
        masks = np.zeros([person_num,h_,w_,13,2])
        person_num_mask = np.zeros([h_,w_]) # 有多少个人的关节在这个区域重叠
        for ii_,(person,middle) in enumerate(keypoint_anno.items()):
            for jj_,(m1,m2) in enumerate(zip(mid_1,mid_2)):
                center1,center2 = middle[m1][:2],middle[m2][:2]
                if middle[m1][2] == 3 or middle[m2][2]==3:
                    # 不再图中的就跳过(初始化为0)
                    continue
                mask = self._PAF(center1,center2,self.ratio)
                person_num_mask  = person_num_mask+ ((mask!=0).sum(axis = 2)>0)
                masks[ii_,:,:,jj_,:] = mask
        masks = masks.sum(axis=0)/(person_num_mask[:,:,np.newaxis,np.newaxis]+0.000001)
        
        self.paf = masks
        return self.paf

    def _PAF(self,center1,center2,ratio):

        '''
        @param center1: 起点
        @param center2: 终点
        @param ratio: 长宽(粗)比

        @return mask: paf, shape(h,w,2)
        '''
        center1,center2 = np.array(center1),np.array(center2)
        h_,w_ = self.re_size
        mask = np.zeros([h_,w_,2])

        length = np.linalg.norm(center1-center2)
        limb_vec_unit = (center2-center1)/(length+0.001)
        limb_perp_unit = np.array([-limb_vec_unit[1],limb_vec_unit[0]])
        width = int(length*ratio/2)+1
        # if width>1:print width

        coords = np.zeros([4,2])
        coords[0] = center1 - limb_perp_unit*width
        coords[1] = center1 + limb_perp_unit*width
        coords[2] = center2 + limb_perp_unit*width
        coords[3] = center2 - limb_perp_unit*width

        # 不要让他超出边界
        for _c in coords:
            if _c[0]<0:_c[0]=0
            if _c[0]>h_-2:_c[0]=h_-2
            if _c[1]<0:_c[1]=0
            if _c[1]>w_-2:_c[1]=w_-2
        x,y = coords[:,0],coords[:,1]
        rr, cc = draw.polygon(x,y)
        mask[rr,cc] = limb_vec_unit
        return mask
    
    def resize(self,img,size,mode='reflect'):
        return transform.resize(img,size,mode=mode)

    def random_crop(self,imgs):
        results = []
        random_size
        for img in imgs:
            result = transform.crop()

    def weight_mask(self):
        mask = np.zeros(self.re_size)
        human_annotations = self.anno['human_annotations']
        for human,anno in human_annotations.items():
            x1,y1,x2,y2 = anno
            mask[x1:x2,y1:y2]=1
        return mask

    def get(self):
        ''' 
        @return img：图片（opt.output_size,opt.output_size,3)
        @return cf:
        @return paf:
        @return mask:
        '''
        img = self.resize(np.asarray(self._img),self.re_size*self.opt.downsample_rate)
        

        paf = self.PAF_map()
        _1,_2,_3,_4 = paf.shape
        paf=paf.reshape(_1,_2,-1)
        cf = self.confidence_map()
        mask = self.weight_mask()[...,np.newaxis]
        size,output_size = np.array(self.size),np.array(self.output_size)
 
        crop_width = ((self.re_size*self.opt.downsample_rate - output_size)/self.opt.downsample_rate).astype(np.int32)
        crop_before = np.floor(crop_width/2).astype(np.int32)
        crop_after = (crop_width.astype(np.int32)-crop_before)
        crop_before = crop_before.tolist()+[0]
        crop_after = crop_after.tolist()+[0]

        paf = self.crop(paf,crop_before,crop_after)
        cf = self.crop(cf,crop_before,crop_after)
        mask = self.crop(mask,crop_before,crop_after)

        crop_after =  [_*self.opt.downsample_rate  for _ in crop_after]
        crop_before = [_*self.opt.downsample_rate for _ in crop_before]

        img = self.crop(img,crop_before,crop_after)



        return (img,cf,paf,mask)

    def rescale(self,img,scale_,mode = 'reflect'):
        return transform.rescale(img,scale_,mode=mode)

    def crop(self,img,before,after):
        return util.crop(img,zip(before,after))

    def flip(self,img,vec=False):
        """
        docstring here
            :param self: 
            :param img: 
            :param vec=False: 
        """   
        if vec:
            img[:,:,::2]*=-1
        else:
            pass
        return img
    def rotate(self,ratio):
        pass


if __name__ == '__main__':
    db = AIChallengerDB('/mnt/7/ai_challenger_keypoint_train_20170909/')
    img, gt = db.get(999)


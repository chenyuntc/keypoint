#coding:utf8
import numpy as np
import json
from scipy.misc import imread
import cv2
from tqdm import tqdm
from pylab import plt
'''
1/右肩，2/右肘，3/右腕，4/左肩，5/左肘，6/左腕，
7/右髋，8/右膝，9/右踝，10/左髋，11/左膝，12/左踝，13/头顶，14/脖子。
'''
part_labels = ['sho_r','elb_r','wri_r','sho_l','elb_l','wri_l','hip_r','kne_r','ank_r','hip_l','kne_l','ank_l','head','neck']
# part_labels = ['nose','eye_l','eye_r','ear_l','ear_r',
#                'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
#                'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r']
part_idx = {b:a for a, b in enumerate(part_labels)}


def show_limb(pred,img_path):
    img = imread(img_path)
    annos =  pred.get('keypoint_annotations', pred.get('keypoint_annos'))
    for human in annos.values():
        draw_limbs(img,human)
    return plt.imshow(img)
#def show_limb2(pred,root):
#    img = root+pred['image_id']+'.jpg'
 #   img = imread(img)
#    annos =  pred.get('keypoint_annotations', pred.get('keypoint_annos'))
 #   for human in annos.values():
  #      draw_limbs(img,human)
  #      break
  #  return plt.imshow(img)
def draw_limbs(inp, pred):
    def link(a, b, color):
        if part_idx[a] < pred.shape[0] and part_idx[b] < pred.shape[0]:
            a = pred[part_idx[a]]
            b = pred[part_idx[b]]
           # if a[2]>0.07 and b[2]>0.07:
        
            if a[2]<3 and b[2]<3 and a[0]*b[0]:
                cv2.line(inp, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 6)

    pred = np.array(pred).reshape(-1, 3)
    bbox = pred[pred[:,2]>0]
    
    a, b, c, d = bbox[:,0][bbox[:,0]>0].min(),  bbox[:,1][bbox[:,1]>0].min(),\
                 bbox[:,0][bbox[:,0]>0].max(),  bbox[:,1][bbox[:,1]>0].max()
    cv2.rectangle(inp, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 5)

    # link('nose', 'eye_l', (255, 0, 0))
    # link('eye_l', 'eye_r', (255, 0, 0))
    # link('eye_r', 'nose', (255, 0, 0))

    # link('eye_l', 'ear_l', (255, 0, 0))
    # link('eye_r', 'ear_r', (255, 0, 0))

    # link('ear_l', 'sho_l', (255, 0, 0))
    # link('ear_r', 'sho_r', (255, 0, 0))
    link('head','neck',(255,0,0))
    link('neck','sho_l',(255,0,0))
    link('neck','sho_r',(255,0,0))
    link('sho_l', 'sho_r', (255, 0, 0))
    link('sho_l', 'hip_l', (0, 255, 0))
    link('sho_r', 'hip_r',(0, 255, 0))
    link('hip_l', 'hip_r', (0, 255, 0))

    link('sho_l', 'elb_l', (0, 0, 255))
    link('elb_l', 'wri_l', (0, 0, 255))

    link('sho_r', 'elb_r', (0, 0, 255))
    link('elb_r', 'wri_r', (0, 0, 255))

    link('hip_l', 'kne_l', (255, 255, 0))
    link('kne_l', 'ank_l', (255, 255, 0))

    link('hip_r', 'kne_r', (255, 255, 0))
    link('kne_r', 'ank_r', (255, 255, 0))

class Config:
    pred = '/data/image/ai_cha/old_del/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
    #pred = '/home/a/code/pytorch/ai_challenge/dc_keypoint/KeyPoint/result/val_result.json'
    file_dir ='/data/image/ai_cha/old_del/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'
    result_img = 'out.jpg'
    limit = 100
opt = Config()

if __name__=='__main__':

    with open(opt.pred) as f:
        data = json.load(f)

    for ii,d in tqdm(enumerate(data[:opt.limit]),total= opt.limit):
        img_path = '%s/%s.jpg' %(opt.file_dir, d['image_id'])
        img = imread(img_path, mode='RGB')
        for human in d['keypoint_annotations'].values():
            draw_limbs(img, human)
        cv2.imwrite('%s.jpg' %ii, img[:,:,::-1])
            



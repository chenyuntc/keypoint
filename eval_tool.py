#coding:utf8
import json
import time
import argparse
import pprint

import numpy as np
from config import opt
from warnings import warn

def parse(kwargs):
    ## 处理配置和参数
    for k, v in kwargs.items():
        if not hasattr(opt, k):
            print("Warning: opt has not attribut %s" % k)
        setattr(opt, k, v)
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'): print(k, getattr(opt, k))


def load_annotations(annos):
    """Convert annotation JSON file."""

    annotations = dict()
    annotations['image_ids'] = set([])
    annotations['annos'] = dict()
    annotations['delta'] = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
                                       0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
                                       0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])
 

    for anno in annos:
        annotations['image_ids'].add(anno['image_id'])
        annotations['annos'][anno['image_id']] = dict()
        annotations['annos'][anno['image_id']]['human_annos'] = anno[
            'human_annotations']
        annotations['annos'][anno['image_id']]['keypoint_annos'] = anno[
            'keypoint_annotations']

    return annotations


def load_predictions(preds):
    """Convert prediction JSON file."""

    predictions = dict()
    predictions['image_ids'] = []
    predictions['annos'] = dict()
    id_set = set([])


    for pred in preds:
        if 'image_id' not in pred.keys():
            warn('There is an invalid annotation info, \
                likely missing key \'image_id\'.')
            continue
        if 'keypoint_annotations' not in pred.keys():
            warn(pred['image_id']+\
                ' does not have key \'keypoint_annotations\'.')
            continue
        image_id = pred['image_id'].split('.')[0]
        if image_id in id_set:
            warn(pred['image_id']+\
                ' is duplicated in prediction JSON file.')
        else:
            id_set.add(image_id)
        predictions['image_ids'].append(image_id)
        predictions['annos'][pred['image_id']] = dict()
        predictions['annos'][pred['image_id']]['keypoint_annos'] = pred[
            'keypoint_annotations']

    return predictions


def compute_oks(anno, predict, delta):
    """Compute oks matrix (size gtN*pN)."""

    anno_count = len(anno['keypoint_annos'].keys())
    predict_count = len(predict.keys())
    oks = np.zeros((anno_count, predict_count))
    if predict_count == 0:return oks.T

    # for every human keypoint annotation
    for i in range(anno_count):
        anno_key = anno['keypoint_annos'].keys()[i]
        anno_keypoints = np.reshape(anno['keypoint_annos'][anno_key], (14, 3))
        visible = anno_keypoints[:, 2] == 1
        bbox = anno['human_annos'][anno_key]
        scale = np.float32((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))
        if np.sum(visible) == 0:
            for j in range(predict_count):
                oks[i, j] = 0
        else:
            # for every predicted human
            for j in range(predict_count):
                predict_key = predict.keys()[j]
                predict_keypoints = np.reshape(predict[predict_key], (14, 3))
                dis = np.sum((anno_keypoints[visible, :2] \
                    - predict_keypoints[visible, :2])**2, axis=1)
                oks[i, j] = np.mean(
                    np.exp(-dis / 2 / delta[visible]**2 / (scale + 1)))
   # if anno_count<50:
    #    print ("oks: ",oks)
    return oks


def keypoint_eval(predictions, annotations):
    """Evaluate predicted_file and return mAP."""

    oks_all = np.zeros((0))
    oks_num = 0
    prediction_id_set = set(predictions['image_ids'])

    # for every annotation in our test/validation set
    for image_id in annotations['image_ids']:
        # if the image in the predictions, then compute oks
        if image_id in prediction_id_set:
            oks = compute_oks(anno=annotations['annos'][image_id], \
                              predict=predictions['annos'][image_id]['keypoint_annos'], \
                              delta=annotations['delta'])
            # view pairs with max OKSs as match ones, add to oks_all
            oks_all = np.concatenate((oks_all, np.max(oks, axis=1)), axis=0)
            # accumulate total num by max(gtN,pN)
            oks_num += np.max(oks.shape)
        else:
            continue
            # otherwise report warning
            warn(image_id + ' is not in the prediction JSON file.')
            # number of humen in ground truth annotations
            gt_n = len(annotations['annos'][image_id]['human_annos'].keys())
            # fill 0 in oks scores
            oks_all = np.concatenate((oks_all, np.zeros((gt_n))), axis=0)
            # accumulate total num by ground truth number
            #oks_num += gt_n

    # compute mAP by APs under different oks thresholds
    #print (oks_num)
    average_precision = []
    for threshold in np.linspace(0.5, 0.95, 10):
        num_thre = np.sum(oks_all > threshold)
        average_precision.append(
            num_thre / np.float32(oks_num))
        #print (num_thre, num_thre / np.float32(oks_num))
    #print (average_precision)
    return np.mean(average_precision),(oks_all,oks_num)



def test(preds,annos):


    #annos = json.load(open(opt.ref)))
    #annotations = load_annotations(annos)

    

    score = keypoint_eval(
        predictions=preds,
        annotations=annos,
        )
    print ('Score: ', '%.8f' % score)



if __name__ == "__main__":
    import fire
    fire.Fire()

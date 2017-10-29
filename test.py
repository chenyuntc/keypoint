# coding:utf8
import copy
import json
import math
import os
from glob import glob

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import ipdb
import cv2 as cv
import fire
import torch as t
from config import opt
from models.layer import get_blocks, make_models
from tqdm import tqdm
import  models
# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[13, 14], [1, 14], [4, 14], [1, 2], [2, 3], [4, 5], [5, 6], [1, 7], [7, 8], [8, 9], [4, 10], [10, 11], [11, 12]]
# the middle joints heatmap correpondence
mapIdx = [[i, i + 1] for i in range(0, 26, 2)]


def store_json(filename, data):
    with open(filename, 'w') as json_file:
        json_file.write(json.dumps(data))


def parse(kwargs):
    # 处理配置和参数
    for k, v in kwargs.iteritems():
        if not hasattr(opt, k):
            print("Warning: opt has not attribut %s" % k)
        setattr(opt, k, v)
    for k, v in opt.__class__.__dict__.iteritems():
        if not k.startswith('__'):
            print(k, getattr(opt, k))


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def main(**kwargs):
    parse(kwargs)
    model = getattr(models, opt.model)(opt).cuda().eval()
    model.load(opt.model_path)
    img_list = glob(opt.val_dir + '*.jpg')
    result = []
    num = 0
    for img in tqdm(img_list):
        if os.path.exists('/tmp/debug22'):
            ipdb.set_trace()
        file_name = img.split('/')[-1][:-4]
        oriImg = cv.imread(img)
        heatmap_avg, paf_avg = get_from_model(model, oriImg)
        all_peaks, peak_counter = get_peaks(heatmap_avg)
        connection_all, special_k = get_links(all_peaks, paf_avg, oriImg)
        subsets = get_subset(all_peaks, connection_all, special_k)
        img_result = {}
        img_result['image_id'] = file_name
        keypoint = {}
        candidate = np.array(
            [item for sublist in all_peaks for item in sublist])
        for ii, sub in enumerate(subsets):
            parts = sub[:-2]
            key_ii = []
            for jj, part in enumerate(parts):
                if part > -1:
                    pp = candidate[int(part)]
                    key_ii += [int(pp[0]), int(pp[1]), 1]
                else:
                    key_ii += [0, 0, 0]
            keypoint["human" + str(ii + 1)] = key_ii
        img_result['keypoint_annotations'] = keypoint
        result.append(img_result)
        num += 1
        if num % 100 == 0:
            store_json(opt.val_result_dir + "val_result.json", result)
    print len(result)
    store_json(opt.val_result_dir + "test_1026_result.json", result)


def flip_paf(paf1, paf2):
    '''
    paf1:原图
    paf2：flip 
    '''
    mid_1 = [13, 1, 4, 1, 2, 4, 5, 1, 7, 8, 4, 10, 11]
    mid_2 = [14, 14, 14, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
    output = np.zeros([paf1.shape[0], paf2.shape[1], 26])
    output[:, :, 0] = paf1[:, :, 0] - paf2[:, ::-1, 0]  #13,14
    output[:, :, 1] = paf1[:, :, 1] + paf2[:, ::-1, 1]  #13,14
    output[:, :, 2] = paf1[:, :, 2] - paf2[:, ::-1, 4]  #1,14
    output[:, :, 3] = paf1[:, :, 3] + paf2[:, ::-1, 6]  #1,14
    output[:, :, 4] = paf1[:, :, 4] - paf2[:, ::-1, 2]  #4,14
    output[:, :, 5] = paf1[:, :, 5] + paf2[:, ::-1, 3]  #4,14
    output[:, :, 6] = paf1[:, :, 6] - paf2[:, ::-1, 10]
    output[:, :, 7] = paf1[:, :, 7] + paf2[:, ::-1, 11]
    output[:, :, 8] = paf1[:, :, 8] - paf2[:, ::-1, 12]
    output[:, :, 9] = paf1[:, :, 9] + paf2[:, ::-1, 13]
    output[:, :, 10] = paf1[:, :, 10] - paf2[:, ::-1, 6]
    output[:, :, 11] = paf1[:, :, 11] + paf2[:, ::-1, 7]
    output[:, :, 12] = paf1[:, :, 12] - paf2[:, ::-1, 8]
    output[:, :, 13] = paf1[:, :, 13] + paf2[:, ::-1, 9]
    output[:, :, 14] = paf1[:, :, 14] - paf2[:, ::-1, 20]
    output[:, :, 15] = paf1[:, :, 15] + paf2[:, ::-1, 21]
    output[:, :, 16] = paf1[:, :, 16] - paf2[:, ::-1, 22]
    output[:, :, 17] = paf1[:, :, 17] + paf2[:, ::-1, 23]
    output[:, :, 18] = paf1[:, :, 18] - paf2[:, ::-1, 24]
    output[:, :, 19] = paf1[:, :, 19] + paf2[:, ::-1, 25]
    output[:, :, 20] = paf1[:, :, 20] - paf2[:, ::-1, 14]
    output[:, :, 21] = paf1[:, :, 21] + paf2[:, ::-1, 15]
    output[:, :, 22] = paf1[:, :, 22] - paf2[:, ::-1, 16]
    output[:, :, 23] = paf1[:, :, 23] + paf2[:, ::-1, 17]
    output[:, :, 24] = paf1[:, :, 24] - paf2[:, ::-1, 18]
    output[:, :, 25] = paf1[:, :, 25] + paf2[:, ::-1, 19]
    return output / 2.0


def flip_heatmap(heatmap1, heatmap2):
    '''
    heatmap1:原图
    heatmap2：flip 
    '''
    output = np.zeros([heatmap1.shape[0], heatmap1.shape[1], 15])
    left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    right = [4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9, 13, 14]
    for ii in range(14):
        output[:, :, ii] = (
            heatmap1[:, :, left[ii] - 1] + heatmap2[:, ::-1, right[ii] - 1]
        ) / 2.0
    output[:, :, 14] = (heatmap1[:, :, 14] + heatmap2[:, ::-1, 14]) / 2.0
    return output


def get_from_model(pose_model, oriImg):

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 26))
    multiplier = [
        x * opt.boxsize * 1.0 / oriImg.shape[0] for x in opt.scale_search
    ]
    mean = np.reshape(
        np.array([0.485, 0.456, 0.406], dtype=np.float32), [3, 1, 1])
    std = np.reshape(
        np.array([0.229, 0.224, 0.225], dtype=np.float32), [3, 1, 1])
    for i in range(len(multiplier)):
        scale = multiplier[i]
        imageToTest = cv.resize(
            oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        imageToTest_flip = imageToTest[:, ::-1, :]
        imageToTest_padded, pad1 = padRightDownCorner(imageToTest, 64, 128)
        imageToTest_padded_flip, pad2 = padRightDownCorner(imageToTest_flip, 64,
                                                        128)

        transposeImage = np.transpose(
            np.float32(imageToTest_padded[:, :, :]), (2, 0, 1)) / 255.0
        transposeImage = (transposeImage - mean) / std
        output1 = pose_model(
            t.autograd.Variable(
                t.from_numpy(transposeImage[np.newaxis, :, :, :]).cuda()))
        heatmap1 = np.transpose(output1[5][0][0:15].data.cpu().numpy(), [1, 2, 0])

        transposeImage = np.transpose(
            np.float32(imageToTest_padded_flip[:, :, :]), (2, 0, 1)) / 255.0
        transposeImage = (transposeImage - mean) / std
        output2 = pose_model(
            t.autograd.Variable(
                t.from_numpy(transposeImage[np.newaxis, :, :, :]).cuda()))
        heatmap2 = np.transpose(output2[5][0,0:15].data.cpu().numpy(), [1, 2, 0])
        #heatmap=flip_heatmap(heatmap1+heatmap2)
        #print (heatmap.shape)

        heatmap1 = cv.resize(
            heatmap1, (0, 0),
            fx=opt.stride,
            fy=opt.stride,
            interpolation=cv.INTER_CUBIC)
        heatmap1 = heatmap1[:imageToTest_padded.shape[0] - pad1[2], :
                            imageToTest_padded.shape[1] - pad1[3], :]
        heatmap1 = cv.resize(
            heatmap1, (oriImg.shape[1], oriImg.shape[0]),
            interpolation=cv.INTER_CUBIC)

        heatmap2 = cv.resize(
            heatmap2, (0, 0),
            fx=opt.stride,
            fy=opt.stride,
            interpolation=cv.INTER_CUBIC)
        heatmap2 = heatmap2[:imageToTest_padded_flip.shape[0] - pad2[2], :
                            imageToTest_padded_flip.shape[1] - pad2[3], :]
        heatmap2 = cv.resize(
            heatmap2, (oriImg.shape[1], oriImg.shape[0]),
            interpolation=cv.INTER_CUBIC)

        heatmap = flip_heatmap(heatmap1, heatmap2)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        paf1 = np.transpose(output1[5][0,15:].data.cpu().numpy(), [1, 2, 0])
        paf2 = np.transpose(output2[5][0,15:].data.cpu().numpy(), [1, 2, 0])
        #print (paf.shape)

        paf1 = cv.resize(
            paf1, (0, 0),
            fx=opt.stride,
            fy=opt.stride,
            interpolation=cv.INTER_CUBIC)
        paf1 = paf1[:imageToTest_padded.shape[0] - pad1[2], :
                    imageToTest_padded.shape[1] - pad1[3], :]
        paf1 = cv.resize(
            paf1, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

        paf2 = cv.resize(
            paf2, (0, 0),
            fx=opt.stride,
            fy=opt.stride,
            interpolation=cv.INTER_CUBIC)
        paf2 = paf2[:imageToTest_padded.shape[0] - pad2[2], :
                    imageToTest_padded.shape[1] - pad2[3], :]
        paf2 = cv.resize(
            paf2, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

        paf = flip_paf(paf1,paf2)
        paf_avg = paf_avg + paf / len(multiplier)
    paf_avg = paf_avg.reshape(paf_avg.shape[0],paf_avg.shape[1],13,2)[:,:,:,::-1].reshape(paf_avg.shape)
    return heatmap_avg, paf_avg


def get_peaks(heatmap_avg):

    all_peaks = []
    peak_counter = 0

    for part in range(14):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]
        #mask 的点分别与它左边 右边 上边 下边的一个点比较  并且〉0.1
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down,
             map > opt.thre1))
        peaks = zip(np.nonzero(peaks_binary)[1],
                    np.nonzero(peaks_binary)[0])  # note reverse
        rett = copy.deepcopy(peaks)
        ll = len(tuple(rett))
        peaks_with_score = [x + (map_ori[x[1], x[0]], ) for x in peaks]
        id = range(peak_counter, peak_counter + ll)
        peaks_with_score_and_id = [
            peaks_with_score[i] + (id[i], ) for i in range(len(id))
        ]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += ll
    return all_peaks, peak_counter


def get_links(all_peaks, paf_avg, oriImg):
    connection_all = []
    special_k = []
    mid_num = 11

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]

        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0:
                        #special_k.append(k)
                        #connection_all.append([])
                        continue

                    vec = np.divide(vec, norm)
                    startend = tuple(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                    #print (tuple(startend)[0])
                    lll = len(tuple(copy.deepcopy(startend)))
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(lll)])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(lll)])
                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(
                        vec_y, vec[1])
                    #论文公式10  求E
                    score_with_dist_prior = sum(
                        score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > opt.thre2)
                                     [0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0

                    if criterion1 and criterion2:
                        connection_candidate.append([
                            i, j, score_with_dist_prior,
                            score_with_dist_prior + candA[i][2] + candB[j][2]
                        ])
                    # print('--------end-----------')
            connection_candidate = sorted(
                connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack(
                        [connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    return connection_all, special_k


def get_subset(all_peaks, connection_all, special_k):
    subset = -1 * np.ones((0, 16))

    candidate = np.array([item for sublist in all_peaks for item in sublist])
    for k in range(len(mapIdx)):

        if k not in special_k:
            # all_peaks id
            partAs = connection_all[k][:, 0]
            # all_peaks id
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j0 = subset_idx[0]
                    if (subset[j0][indexB] != partBs[i]):
                        subset[j0][indexB] = partBs[i]
                        subset[j0][-1] += 1
                        subset[j0][-2] += candidate[partBs[i].astype(
                            int), 2] + connection_all[k][i][2]
                    else:
                        subset[j0][indexA] = partAs[i]
                        subset[j0][-1] += 1
                        subset[j0][-2] += candidate[partAs[i].astype(
                            int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) +
                                  (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(
                            int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 13:
                    row = -1 * np.ones(16)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(
                        int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 5 or subset[i][-2] / subset[i][-1] < 0.5:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    return subset


if __name__ == "__main__":
    fire.Fire()

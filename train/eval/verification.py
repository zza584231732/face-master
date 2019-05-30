# -*-coding:utf-8-*-
"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
import cv2
import math
import datetime
import pickle
#import cPickle as pickle    #python 2
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_image
import pdb


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    # thresholds: np.arange(0, 4, 0.01)
    # embeddings1:(6000L, 512L)
    # embeddings2:(6000L, 512L)
    # actual_issame:len(actual_issame)==6000

    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])  # 6000L

    nrof_thresholds = len(thresholds)  # 400L
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))  # (10L,400L)
    fprs = np.zeros((nrof_folds, nrof_thresholds))  # (10L,400L)
    accuracy = np.zeros((nrof_folds))  # (10L,)
    indices = np.arange(nrof_pairs)  # np.arange(6000)

    if pca == 0:
        # 求欧氏距离平方
        diff = np.subtract(embeddings1, embeddings2)  # (6000L,512L)
        dist = np.sum(np.square(diff), 1)  # (6000L,)

    # A = k_fold.split(indices)
    # for _ in xrange(10):
    #   B = A.next()
    #   print(len(B[0]))
    #   print(len(B[1]))
    #   print
    # sys.exit()

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):  # k_fold.split(indices): 可next10次的生成器
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]  # 选择5400个作为训练集
            embed2_train = embeddings2[train_set]  # 选择5400个作为训练集
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # 遍历thresholds, 发现fold最好的threshold
        acc_train = np.zeros((nrof_thresholds))  # (400L,)
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])

        best_threshold_index = np.argmax(acc_train)

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])

        print('thresholds[best_threshold_index]:',thresholds[best_threshold_index])
        with open('result_com.txt', 'a') as f:
            f.writelines(str(thresholds[best_threshold_index]))
            f.writelines('\n')
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    print(accuracy)
    # sys.exit()

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    #
    predict_issame = np.less(dist, threshold)  # 前者小于后者时返回True,参见 np.info(np.less)

    # True positive, False positive, True Negtive, False Negtive
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # print('tp:',tp)
    # print('fp',fp)
    # print('tn',tn)
    # print('fn',fn)
    # print('dist.size',dist.size)
    # sys.exit()

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)  # 查准率tr
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)  # 假正例率
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    # thresholds: np.arange(0, 4, 0.001)
    # embeddings1: (6000L, 512L)
    # embeddings2: (6000L, 512L)
    # actual_issame: (6000L,)
    # far_target: 0.001

    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])  # 6000
    nrof_thresholds = len(thresholds)  # 4000
    k_fold = LFold(n_splits=nrof_folds, shuffle=True)  # 更改了

    val = np.zeros(nrof_folds)  # (10L,)
    far = np.zeros(nrof_folds)  # (10L,)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)  # 欧氏距离平方

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):  # next10次
        # print(train_set)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)  # (4000L,)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    # threshold: scalar
    # print('dist.shape',dist.shape)
    # print('actual_issame.shape',actual_issame.shape)
    # print('*'*20)

    predict_issame = np.less(dist, threshold)  # 小于阈值的都为True
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))  # 把对的预测对的的数量(把一样的图像预测为一样) #TP
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))  # 把不一致的预测为一致, #FP

    # print('true_accept:',true_accept)
    # print('false_accept:',false_accept)
    # print('*'*20)
    # print('threshold',threshold)
    # if threshold == 3.99:
    #     pdb.set_trace()
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print('actual_issame:',actual_issame)
    # sys.exit(0)

    # print('n_same:',n_same)
    # print('n_diff:',n_diff)
    # print('*'*20)

    # print('true_accept, false_accept:',true_accept, false_accept)
    # print('n_same, n_diff:',n_same, n_diff)
    # pdb.set_trace()
    if n_same == 0:
        n_same = 1e-4

    val = float(true_accept) / float(n_same)  # TP/所有正例,即TPR
    if n_diff == 0:
        n_diff = 1e-4
    far = float(false_accept) / float(n_diff)  # FP/所有反例,即FPR

    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # embeddings: (12000L, 512L)
    # actual_issame: (6000L,)

    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.001)
    embeddings1 = embeddings[0::2]  # 从0开始,隔一行取一个,如:0,2,4,...等,shape:(6000L, 512L)
    embeddings2 = embeddings[1::2]  # 从1开始,隔一行取一个,如:1,3,5,...等,shape:(6000L, 512L)

    print('embeddings.shape:',embeddings.shape)
    print('embeddings1.shape:',embeddings1.shape)
    print('embeddings2.shape:',embeddings2.shape)
    # sys.exit()

    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)

    thresholds = np.arange(0, 4, 0.001)

    # val, val_std, far = 0,0,0
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)

    with open('result.txt', 'a') as f:
        f.writelines(str(accuracy))
        f.writelines('\n')
    return tpr, fpr, accuracy, val, val_std, far


def load_bin(path, image_size):
    # path为: ../datasets/faces_ms1m_112x112/lfw.bin
    # image_size为: [112, 112]

    #bins, issame_list = pickle.load(open(path, 'rb'),encoding='bytes')
    bins, issame_list = pickle.load(open(path, 'rb'))# bins存储图片对数据,issame_list存储标签
    print('len(bins):',len(bins))
    # sys.exit()
    # print('issame_list:',issame_list)
    # sys.exit()
    # pdb.set_trace()
    data_list = []
    for _ in [0, 1]:
        data = nd.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))  # (12000L, 3L, 112L, 112L)的NDArray
        data_list.append(data)

    for i in range(len(issame_list) * 2):  # 12000
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        # print(img.shape)
        img = mx.image.imresize(img,96,112)
        #img = mx.image.imresize(img,96,112)
        # print(img.shape)
        img = nd.transpose(img, axes=(2, 0, 1))

        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)  # 水平翻转

            data_list[flip][i][:] = img
        if i % 1000 == 0:
            print('loading bin', i)

    print('issame_list[0]:',issame_list[0])
    # sys.exit()
    # data_list: 两个元素,每个元素为(12000L, 3L, 112L, 112L)的NDArray
    # issame_list: 长度为6000的列表,每个元素为bool
    return (data_list, issame_list)


def test(data_set, mx_model, batch_size, nfolds=10, data_extra=None, label_shape=None):
    # data_set 存放数据和标签的元组
    # print('mx_model:',mx_model)
    # sys.exit(0)
    # pdb.set_trace()
    print('testing verification..')
    data_list = data_set[0]  # data_list: 两个元素,每个元素为(12000L, 3L, 112L, 112L)的NDArray
    issame_list = data_set[1]  # issame_list: 长度为6000的列表,每个元素为bool
    model = mx_model
    embeddings_list = []
    if data_extra is not None:
        _data_extra = nd.array(data_extra)
    time_consumed = 0.0
    if label_shape is None:
        _label = nd.ones((batch_size,))  # (32L,)的 1
    else:
        _label = nd.ones(label_shape)

    # print('len(data_list);',len(data_list))
    # sys.exit(0)
    # pdb.set_trace()
    for i in range(len(data_list)):  # 2
        data = data_list[i]  # (12000L, 3L, 112L, 112L)的NDArray

        embeddings = None
        ba = 0
        while ba < data.shape[0]:  # 12000
            bb = min(ba + batch_size, data.shape[0])  # 32,64,96,128,160,192...
            count = bb - ba  # 32,32,...

            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)  # (32L, 3L, 112L, 112L)
            # print('_data.shape, _label.shape:',_data.shape, _label.shape) #(64, 3, 112, 112) (64,)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra), label=(
                _label,))  # DataBatch: data shapes: [(32L, 3L, 112L, 112L)] label shapes: [(32L,)]

            model.forward(db, is_train=False)
            net_out = model.get_outputs()  # len(net_out)==1,其中的元素为(32L, 512L)的NDArray

            # _arg, _aux = model.get_params()
            # __arg = {}
            # for k,v in _arg.iteritems():
            #  __arg[k] = v.as_in_context(_ctx)
            # _arg = __arg
            # _arg["data"] = _data.as_in_context(_ctx)
            # _arg["softmax_label"] = _label.as_in_context(_ctx)
            # for k,v in _arg.iteritems():
            #  print(k,v.context)
            # exe = sym.bind(_ctx, _arg ,args_grad=None, grad_req="null", aux_states=_aux)
            # exe.forward(is_train=False)
            # net_out = exe.outputs
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            # print('_embeddings.shape:',_embeddings.shape)
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]  # (12000, 512)
            ba = bb
        embeddings_list.append(embeddings)  # 循环执行完长度为2,保存imgs和imgs_flip
    # pdb.set_trace()
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):  # 12000
            _em = embed[i]  # 512
            _norm = np.linalg.norm(_em)  # 元素平方加和开根号

            _xnorm += _norm
            _xnorm_cnt += 1

    _xnorm /= _xnorm_cnt

    acc1 = 0.0
    std1 = 0.0

    embeddings = embeddings_list[0] + embeddings_list[1]  # 两者element-wise元素加和,shape:(12000, 512)
    # if np.isnan(embeddings).any():
    #     print('有缺失值')
    #     embeddings_inf = np.isnan(embeddings)
    #     embeddings[embeddings_inf] = 0.0
    # #
    # if np.isfinite(embeddings).any():
    #     print('有无穷大')
    #     embeddings_inf = np.isinf(embeddings)
    #     embeddings[embeddings_inf] = 0.0
    # 标准化
    embeddings = sklearn.preprocessing.normalize(embeddings)  # (12000, 512)

    print('infer time', time_consumed)

    # accuracy:(10L,),存放10折交叉验证的10次结果
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)

    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


def test_badcase(data_set, mx_model, batch_size, name='', data_extra=None, label_shape=None):
    print('testing verification badcase..')
    data_list = data_set[0]
    issame_list = data_set[1]
    model = mx_model
    embeddings_list = []
    if data_extra is not None:
        _data_extra = nd.array(data_extra)
    time_consumed = 0.0
    if label_shape is None:
        _label = nd.ones((batch_size,))
    else:
        _label = nd.ones(label_shape)
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            # print(_data.shape, _label.shape)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra), label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    thresholds = np.arange(0, 4, 0.01)
    actual_issame = np.asarray(issame_list)
    nrof_folds = 10
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    data = data_list[0]

    pouts = []
    nouts = []

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        # print('train_set:',train_set)
        # print('train_set.__class__:',train_set.__class__)
        for threshold_idx, threshold in enumerate(thresholds):
            p2 = dist[train_set]
            p3 = actual_issame[train_set]
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, p2, p3)
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])
        best_threshold = thresholds[best_threshold_index]
        for iid in test_set:
            ida = iid * 2
            idb = ida + 1
            asame = actual_issame[iid]
            _dist = dist[iid]
            violate = _dist - best_threshold
            if not asame:
                violate *= -1.0
            if violate > 0.0:
                imga = data[ida].asnumpy().transpose((1, 2, 0))[..., ::-1]  # to bgr
                imgb = data[idb].asnumpy().transpose((1, 2, 0))[..., ::-1]
                print('imga.shape, imgb.shape, violate, asame, _dist:',imga.shape, imgb.shape, violate, asame, _dist)
                if asame:
                    pouts.append((imga, imgb, _dist, best_threshold, ida))
                else:
                    nouts.append((imga, imgb, _dist, best_threshold, ida))

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    acc = np.mean(accuracy)
    pouts = sorted(pouts, key=lambda x: x[2], reverse=True)
    nouts = sorted(nouts, key=lambda x: x[2], reverse=False)
    print(len(pouts), len(nouts))
    print('acc', acc)
    gap = 10
    image_shape = (112, 224, 3)
    out_dir = "./badcases"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if len(nouts) > 0:
        threshold = nouts[0][3]
    else:
        threshold = pouts[-1][3]

    for item in [(pouts, 'positive(false_negative).png'), (nouts, 'negative(false_positive).png')]:
        cols = 4
        rows = 8000
        outs = item[0]
        if len(outs) == 0:
            continue
        # if len(outs)==9:
        #  cols = 3
        #  rows = 3

        _rows = int(math.ceil(len(outs) / cols))
        rows = min(rows, _rows)
        hack = {}

        if name.startswith('cfp') and item[1].startswith('pos'):
            hack = {0: 'manual/238_13.jpg.jpg', 6: 'manual/088_14.jpg.jpg', 10: 'manual/470_14.jpg.jpg',
                    25: 'manual/238_13.jpg.jpg', 28: 'manual/143_11.jpg.jpg'}

        filename = item[1]
        if len(name) > 0:
            filename = name + "_" + filename
        filename = os.path.join(out_dir, filename)
        img = np.zeros((image_shape[0] * rows + 20, image_shape[1] * cols + (cols - 1) * gap, 3), dtype=np.uint8)
        img[:, :, :] = 255
        text_color = (0, 0, 153)
        text_color = (255, 178, 102)
        text_color = (153, 255, 51)
        for outi, out in enumerate(outs):
            row = outi // cols
            col = outi % cols
            if row == rows:
                break
            imga = out[0].copy()
            imgb = out[1].copy()
            if outi in hack:
                idx = out[4]
                print('noise idx', idx)
                aa = hack[outi]
                imgb = cv2.imread(aa)
                # if aa==1:
                #  imgb = cv2.transpose(imgb)
                #  imgb = cv2.flip(imgb, 1)
                # elif aa==3:
                #  imgb = cv2.transpose(imgb)
                #  imgb = cv2.flip(imgb, 0)
                # else:
                #  for ii in xrange(2):
                #    imgb = cv2.transpose(imgb)
                #    imgb = cv2.flip(imgb, 1)
            dist = out[2]
            _img = np.concatenate((imga, imgb), axis=1)
            k = "%.3f" % dist
            # print(k)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(_img, k, (80, image_shape[0] // 2 + 7), font, 0.6, text_color, 2)
            # _filename = filename+"_%d.png"%outi
            # cv2.imwrite(_filename, _img)
            img[row * image_shape[0]:(row + 1) * image_shape[0],
            (col * image_shape[1] + gap * col):((col + 1) * image_shape[1] + gap * col), :] = _img
        # threshold = outs[0][3]
        font = cv2.FONT_HERSHEY_SIMPLEX
        k = "threshold: %.3f" % threshold
        cv2.putText(img, k, (img.shape[1] // 2 - 70, img.shape[0] - 5), font, 0.6, text_color, 2)
        cv2.imwrite(filename, img)


def dumpR(data_set, mx_model, batch_size, name='', data_extra=None, label_shape=None):
    print('dump verification embedding..')
    data_list = data_set[0]
    issame_list = data_set[1]
    print('issame_list:',issame_list)
    model = mx_model
    embeddings_list = []
    if data_extra is not None:
        _data_extra = nd.array(data_extra)
    time_consumed = 0.0
    if label_shape is None:
        _label = nd.ones((batch_size,))
    else:
        _label = nd.ones(label_shape)
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            # print('_data.shape, _label.shape:',_data.shape, _label.shape)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra), label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    print('embeddings:',embeddings)
    embeddings = sklearn.preprocessing.normalize(embeddings)
    actual_issame = np.asarray(issame_list)
    outname = os.path.join('temp.bin')
    with open(outname, 'wb') as f:
        pickle.dump((embeddings, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--data-dir', default='../../dataset/test/', help='')
    parser.add_argument('--model', default='../../my_model/mface-model,0019', help='path to load model.')
    parser.add_argument('--target', default='lfw,cfp_ff,cfp_fp,agedb_30', help='test targets.')
    parser.add_argument('--gpu', default=None, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=1, type=int, help='')
    parser.add_argument('--max', default='', type=str, help='')
    parser.add_argument('--mode', default=2, type=int, help='')
    parser.add_argument('--nfolds', default=10, type=int, help='')
    args = parser.parse_args()

    prop = face_image.load_property(args.data_dir)  # prop存放的是property文件的内容,easydict类

    image_size = prop.image_size  # 112x112

    ctx = mx.gpu(args.gpu)  # context
    nets = []
    vec = args.model.split(',')  # 模型列表,如: ['../models/model-r50-am-lfw/']

    prefix = args.model.split(',')[0]  # 模型所在文件,如: ../models/model-r50-am-lfw/

    epochs = []
    if len(vec) == 1:
        pdir = os.path.dirname(prefix)  # ../models/model-r50-am-lfw

        for fname in os.listdir(pdir):
            if not fname.endswith('.params'):
                continue
            _file = os.path.join(pdir, fname)
            print(' _file:',_file)
            if _file.startswith(prefix):
                print('_file.startswith(prefix):yes')
                epoch = int(fname.split('.')[0].split('-')[1])
                print(epoch)
                epochs.append(epoch)
        epochs = sorted(epochs, reverse=True)
        if len(args.max) > 0:
            _max = [int(x) for x in args.max.split(',')]
            assert len(_max) == 2
            if len(epochs) > _max[1]:
                epochs = epochs[_max[0]:_max[1]]

    else:
        epochs = [int(x) for x in vec[1].split('|')]
    print(epochs) #[0]
    print('model number', len(epochs))

    '''加载模型'''
    print('Loading Model...')
    time0 = datetime.datetime.now()
    for epoch in epochs:
        print('loading', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        # arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, label_names=None)
        model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        nets.append(model)
    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds())

    ver_list = []  # 存储数据和标签
    ver_name_list = []  # verification 数据集列表
    # pdb.set_trace()
    for name in args.target.split(','):  # 遍历测试文件
        path = os.path.join(args.data_dir, name + ".bin")  # 模型所在文件: ../datasets/lfw.bin

        if os.path.exists(path):
            print('loading.. ', name)
            data_set = load_bin(path, image_size)  # data_set: (data_list, issame_list)
            ver_list.append(data_set)
            ver_name_list.append(name)
    # sys.exit()
    # print(ver_list)
    if args.mode == 0:
        for i in range(len(ver_list)):
            results = []
            for model in nets:
                acc1, std1, acc2, std2, xnorm, embeddings_list = test(ver_list[i], model, args.batch_size, args.nfolds)
                print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
                print('[%s]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], acc1, std1))
                print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], acc2, std2))
                results.append(acc2)
            with open('result_com.txt', 'a') as f:
                f.writelines(str(len(ver_list[0][1])))
                f.writelines('\n')
                f.writelines('*' * 50 + str(args.target) + '*' * 50 + '\n')
            print('Max of [%s] is %1.5f' % (ver_name_list[i], np.max(results)))
    elif args.mode == 1:
        model = nets[0]
        test_badcase(ver_list[0], model, args.batch_size, args.target)
    else:
        print('dump')
        model = nets[0]
        dumpR(ver_list[0], model, args.batch_size, args.target)

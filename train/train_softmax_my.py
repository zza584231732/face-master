# -*-coding:utf-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import sys
import math
import random
import logging
import pickle
import numpy as np
from image_iter import FaceImageIter
from image_iter import FaceImageIterList
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image

sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fresnet
import finception_resnet_v2
import fmobilenet
import fmobilenetv2
import fmobilefacenet
import fxception
import fdensenet
import fdpn
import fnasnet
import spherenet
import verification
# import mobilenetv3
# import fmobilenetv3
import sklearn

# sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
# import center_loss

Log = "./log.txt"

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# 添加写日志类
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger(Log)  # 保存日志文件

args = None


class AccMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(AccMetric, self).__init__(
            'acc', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        self.count += 1
        preds = [preds[1]]  # use softmax output
        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32').flatten()
            label = label.asnumpy()
            if label.ndim == 2:
                label = label[:, 0]
            label = label.astype('int32').flatten()
            assert label.shape == pred_label.shape
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)


class LossValueMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(LossValueMetric, self).__init__(
            'lossvalue', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []

    def update(self, labels, preds):
        loss = preds[-1].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1.0
        gt_label = preds[-2].asnumpy()
        print('gt_label:',gt_label)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--data-dir', default='../dataset/', help='training set directory')#../dataset/faces_vgg_112x112,../dataset/train_extface
    parser.add_argument('--prefix', default='../my_model/mface-B-model', help='directory to save model.')
    parser.add_argument('--pretrained', default='', help='pretrained model to load')#../my_model/mface-model,0126,../my_model/mobileface1-model,0200,../my_model/mobileface2-model,0020
    parser.add_argument('--ckpt', type=int, default=1,
                        help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
    parser.add_argument('--loss-type', type=int, default=4, help='loss type')
    parser.add_argument('--verbose', type=int, default=3000,
                        help='do verification testing and model saving every verbose batches')
    parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
    parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
    parser.add_argument('--network', default='y1', help='specify network')
    parser.add_argument('--version-se', type=int, default=1, help='whether to use se in network')
    parser.add_argument('--version-input', type=int, default=1, help='network input config')
    parser.add_argument('--version-output', type=str, default='GNAP', help='network embedding output config')
    parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config')
    parser.add_argument('--version-act', type=str, default='relu', help='network activation config')
    parser.add_argument('--use-deformable', type=int, default=0, help='use deformable cnn in network')
    parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
    parser.add_argument('--fc7-lr-mult', type=float, default=1.0, help='lr mult for fc7')
    parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument('--per-batch-size', type=int, default=64, help='batch size in each context')
    parser.add_argument('--margin-m', type=float, default=0.5, help='margin for loss')
    parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
    parser.add_argument('--margin-a', type=float, default=1.0, help='')
    parser.add_argument('--margin-b', type=float, default=0.0, help='')
    parser.add_argument('--easy-margin', type=int, default=1, help='')
    parser.add_argument('--margin', type=int, default=4, help='margin for sphere')
    parser.add_argument('--beta', type=float, default=1000, help='param for sphere')
    parser.add_argument('--beta_min', type=float, default=5.0, help='param for sphere')
    parser.add_argument('--beta-freeze', type=int, default=0, help='param for sphere')
    parser.add_argument('--gamma', type=float, default=0.12, help='param for sphere')
    parser.add_argument('--power', type=float, default=1.0, help='param for sphere')
    parser.add_argument('--scale', type=float, default=0.9993, help='param for sphere')
    parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
    parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
    parser.add_argument('--target', type=str, default='faces_vgg_112x112/lfw,faces_vgg_112x112/cfp_fp,faces_vgg_112x112/agedb_30,faces_vgg_112x112/ext_test', help='verification targets')#lfw,cfp_fp,agedb_30,,extface_test,faces_vgg_112x112
    args = parser.parse_args()
    return args


def get_symbol(args, arg_params, aux_params):
    # data_shape = (args.image_channel, args.image_h, args.image_w)  # (3L,112L,112L)
    # image_shape = ",".join([str(x) for x in data_shape]) #3,112,112

    # margin_symbols = []
    print('***network: ', args.network)  # r100

    if args.network[0] == 'd':  # densenet
        embedding = fdensenet.get_symbol(args.emb_size, args.num_layers,
                                         version_se=args.version_se, version_input=args.version_input,
                                         version_output=args.version_output, version_unit=args.version_unit)
    elif args.network[0] == 'm':  # mobilenet
        print('init mobilenet', args.num_layers)
        if args.num_layers == 1:
            embedding = fmobilenet.get_symbol(args.emb_size,
                                              version_se=args.version_se, version_input=args.version_input,
                                              version_output=args.version_output, version_unit=args.version_unit)
        else:
            embedding = fmobilenetv2.get_symbol(args.emb_size)
    # elif args.network[0] == 'v':
    #     print('init MobileNet-V3', args.num_layers)
    #     embedding = fmobilenetv3.get_symbol(args.emb_size)
    elif args.network[0] == 'i':  # inception-resnet-v2
        print('init inception-resnet-v2', args.num_layers)
        embedding = finception_resnet_v2.get_symbol(args.emb_size,
                                                    version_se=args.version_se, version_input=args.version_input,
                                                    version_output=args.version_output, version_unit=args.version_unit)
    elif args.network[0] == 'x':
        print('init xception', args.num_layers)
        embedding = fxception.get_symbol(args.emb_size,
                                         version_se=args.version_se, version_input=args.version_input,
                                         version_output=args.version_output, version_unit=args.version_unit)
    elif args.network[0] == 'p':
        print('init dpn', args.num_layers)
        embedding = fdpn.get_symbol(args.emb_size, args.num_layers,
                                    version_se=args.version_se, version_input=args.version_input,
                                    version_output=args.version_output, version_unit=args.version_unit)
    elif args.network[0] == 'n':
        print('init nasnet', args.num_layers)
        embedding = fnasnet.get_symbol(args.emb_size)
    elif args.network[0] == 's':
        print('init spherenet', args.num_layers)
        embedding = spherenet.get_symbol(args.emb_size, args.num_layers)
    elif args.network[0] == 'y':
        print('init mobilefacenet', args.num_layers)
        embedding = fmobilefacenet.get_symbol(args.emb_size, bn_mom=args.bn_mom, version_output=args.version_output)
    else:  # 执行resnet
        print('init resnet, 层数: ', args.num_layers)
        embedding = fresnet.get_symbol(args.emb_size,
                                       args.num_layers,
                                       version_se=args.version_se,
                                       version_input=args.version_input,
                                       version_output=args.version_output,
                                       version_unit=args.version_unit,
                                       version_act=args.version_act)
    # get_symbol
    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label
    # extra_loss = None
    # 重新定义fc7的权重
    _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=args.fc7_lr_mult,
                                 wd_mult=args.fc7_wd_mult)
    if args.loss_type == 0:  # softmax
        _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
        fc7 = mx.sym.FullyConnected(data=embedding, weight=_weight, bias=_bias, num_hidden=args.num_classes, name='fc7')
    elif args.loss_type == 1:  # sphere
        print('*******'*10)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        fc7 = mx.sym.LSoftmax(data=embedding, label=gt_label, num_hidden=args.num_classes,
                              weight=_weight,
                              beta=args.beta, margin=args.margin, scale=args.scale,
                              beta_min=args.beta_min, verbose=1000, name='fc7')
    elif args.loss_type == 2:  # CosineFace
        s = args.margin_s
        m = args.margin_m
        assert (s > 0.0)
        assert (m > 0.0)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s

        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                    name='fc7')
        s_m = s * m
        gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=s_m,
                                    off_value=0.0)  # onehot两个值最大值s_m,最小值0.0
        fc7 = fc7 - gt_one_hot
    elif args.loss_type == 4:  # ArcFace
        s = args.margin_s  # 参数s， 64
        m = args.margin_m  # 参数m， 0.5

        assert s > 0.0
        assert m >= 0.0
        assert m < (math.pi / 2)
        # pdb.set_trace()
        # 权重归一化
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')  # shape = [(4253, 512)]
        # 特征归一化，并放大到 s*x
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                    name='fc7')  # args.num_classes:8631

        zy = mx.sym.pick(fc7, gt_label, axis=1)  # fc7每一行找出gt_label对应的值, 即s*cos_t

        cos_t = zy / s  # 网络输出output = s*x/|x|*w/|w|*cos(theta), 这里将输出除以s，得到实际的cos值，即cos（theta)
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = math.sin(math.pi - m) * m  # sin(pi-m)*m = sin(m) * m  0.2397
        # threshold = 0.0
        threshold = math.cos(math.pi - m)  # 这个阈值避免theta+m >= pi, 实际上threshold<0 -cos(m)    -0.8775825618903726
        if args.easy_margin:  # 将0作为阈值，得到超过阈值的索引
            cond = mx.symbol.Activation(data=cos_t, act_type='relu')
        else:
            cond_v = cos_t - threshold  # 将负数作为阈值
            cond = mx.symbol.Activation(data=cond_v, act_type='relu')
        body = cos_t * cos_t  # cos_t^2 + sin_t^2 = 1
        body = 1.0 - body
        sin_t = mx.sym.sqrt(body)
        new_zy = cos_t * cos_m  # cos(t+m) = cos(t)cos(m) - sin(t)sin(m)
        b = sin_t * sin_m
        new_zy = new_zy - b
        new_zy = new_zy * s  # s*cos(t + m)
        if args.easy_margin:
            zy_keep = zy  # zy_keep为zy，即s*cos(theta)
        else:
            zy_keep = zy - s * mm  # zy-s*sin(m)*m = s*cos(t)- s*m*sin(m)
        new_zy = mx.sym.where(cond, new_zy,
                              zy_keep)  # cond中>0的保持new_zy=s*cos(theta+m)不变，<0的裁剪为zy_keep= s*cos(theta) or s*cos(theta)-s*m*sin(m)

        diff = new_zy - zy
        diff = mx.sym.expand_dims(diff, 1)
        gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
        body = mx.sym.broadcast_mul(gt_one_hot, diff)  # 对应yi处为new_zy - zy
        fc7 = fc7 + body  # 对应yi处，fc7=zy + (new_zy - zy) = new_zy，即cond中>0的为s*cos(theta+m)，<0的裁剪为s*cos(theta) or s*cos(theta)-s*m*sin(m)
    elif args.loss_type == 5:
        s = args.margin_s
        m = args.margin_m
        assert s > 0.0
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                    name='fc7')
        if args.margin_a != 1.0 or args.margin_m != 0.0 or args.margin_b != 0.0:
            if args.margin_a == 1.0 and args.margin_m == 0.0:
                s_m = s * args.margin_b
                gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=s_m, off_value=0.0)
                fc7 = fc7 - gt_one_hot
            else:
                zy = mx.sym.pick(fc7, gt_label, axis=1)
                cos_t = zy / s
                t = mx.sym.arccos(cos_t)
                if args.margin_a != 1.0:
                    t = t * args.margin_a
                if args.margin_m > 0.0:
                    t = t + args.margin_m
                body = mx.sym.cos(t)
                if args.margin_b > 0.0:
                    body = body - args.margin_b
                new_zy = body * s
                diff = new_zy - zy
                diff = mx.sym.expand_dims(diff, 1)
                gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
                body = mx.sym.broadcast_mul(gt_one_hot, diff)
                fc7 = fc7 + body
    out_list = [mx.symbol.BlockGrad(embedding)]
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label=gt_label, name='softmax', normalization='valid')
    out_list.append(softmax)

    out = mx.symbol.Group(out_list)
    # print(out)
    # sys.exit()
    return (out, arg_params, aux_params)


def train_net(args):
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()  # 0,使用第一块GPU

    if len(cvd) > 0:
        for i in range(len(cvd.split(','))):
            ctx.append(mx.gpu(i))  # 讲GPU context添加到ctx,ctx = [gpu(0)]

    if len(ctx) == 0:
        ctx = [mx.cpu()]
        print('use cpu')
    else:
        print('gpu num:', len(ctx))  # 使用了gpu

    prefix = args.prefix  # ../model-r100
    prefix_dir = os.path.dirname(prefix)  # ..

    if not os.path.exists(prefix_dir):  # 未执行
        os.makedirs(prefix_dir)

    end_epoch = args.end_epoch  # 100 000

    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])

    print('num_layers', args.num_layers)  # 100

    if args.per_batch_size == 0:
        args.per_batch_size = 128
    args.batch_size = args.per_batch_size * args.ctx_num  # 10

    args.rescale_threshold = 0
    args.image_channel = 3

    os.environ['BETA'] = str(args.beta)  # 1000.0,参见Arcface公式(6)，退火训练的lambda

    data_dir_list = args.data_dir.split(',')
    print('data_dir_list: ', data_dir_list)

    data_dir = data_dir_list[0]

    # 加载数据集属性
    prop = face_image.load_property(data_dir)
    args.num_classes = prop.num_classes
    image_size = prop.image_size
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)
    print('num_classes: ', args.num_classes)

    path_imgrec = os.path.join(data_dir, "train8631_list.rec")

    if args.loss_type == 1 and args.num_classes > 20000:  # sphereface
        args.beta_freeze = 5000
        args.gamma = 0.06

    print('***Called with argument:', args)

    data_shape = (args.image_channel, image_size[0], image_size[1])  # (3L,112L,112L)

    mean = None

    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd  # weight decay = 0.0005
    base_mom = args.mom  # 动量:0.9

    if len(args.pretrained) == 0:
        arg_params = None
        aux_params = None
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    else:
        vec = args.pretrained.split(',')  # ['../models/model-r50-am-lfw/model', '0000']
        print('***loading', vec)
        _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
        # print('sym[1]:',sym[1])
        # # mx.viz.plot_network(sym[1]).view() #可视化
        # sys.exit()
    if args.network[0] == 's':  # spherenet
        data_shape_dict = {'data': (args.per_batch_size,) + data_shape}
        spherenet.init_weights(sym, data_shape_dict, args.num_layers)

    # label_name = 'softmax_label'
    # label_shape = (args.batch_size,)
    model = mx.mod.Module(
        context=ctx,
        symbol=sym,
    )

    # print('args.batch_size:',args.batch_size)
    # print('data_shape:',data_shape)
    # print('path_imgrec:',path_imgrec)
    # print('args.rand_mirror:',args.rand_mirror)
    # print('mean:',mean)
    # print('args.cutoff:',args.cutoff)
    # sys.exit()

    train_dataiter = FaceImageIter(
        batch_size=args.batch_size,
        data_shape=data_shape,  # (3L,112L,112L)
        path_imgrec=path_imgrec,  # train.rec
        shuffle=True,
        rand_mirror=args.rand_mirror,  # 1
        mean=mean,
        cutoff=args.cutoff,  # 0
    )

    if args.loss_type < 10:
        _metric = AccMetric()
    else:
        _metric = LossValueMetric()
    # 创建一个评价指标
    eval_metrics = [mx.metric.create(_metric)]

    if args.network[0] == 'r' or args.network[0] == 'y' or args.network[0] == 'v':
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2)  # resnet style  mobilefacenet
    elif args.network[0] == 'i' or args.network[0] == 'x':
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)  # inception
    else:
        initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0 / args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd,
                        rescale_grad=_rescale)  # 多卡训练的话，rescale_grad将总的结果分开
    # opt = optimizer.Adam(learning_rate=base_lr, wd=base_wd,rescale_grad=_rescale)
    som = 64
    # 回调函数，用来阶段性显示训练速度和准确率
    _cb = mx.callback.Speedometer(args.batch_size, som)

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        path = os.path.join(data_dir, name + ".bin")
        if os.path.exists(path):
            data_set = verification.load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
            print('ver', name)

    def ver_test(nbatch):
        results = []
        for i in range(len(ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10,
                                                                               None, None)
            print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
            print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
            results.append(acc2)
        return results


    highest_acc = [0.0, 0.0]  # lfw and target
    # for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    if len(args.lr_steps) == 0:
        lr_steps = [30000, 40000, 50000]
        if args.loss_type >= 1 and args.loss_type <= 7:
            lr_steps = [10000, 20000, 40000, 70000, 100000, 150000]
        # 单GPU，去掉p
        # p = 512.0/args.batch_size
        for l in range(len(lr_steps)):
            # lr_steps[l] = int(lr_steps[l]*p)
            lr_steps[l] = int(lr_steps[l])
    else:
        lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)

    def _batch_callback(param):
        # global global_step

        mbatch = global_step[0]
        global_step[0] += 1
        for _lr in lr_steps:
            if mbatch == args.beta_freeze + _lr:
                opt.lr *= 0.1
                print('lr change to', opt.lr)
                break

        _cb(param)
        if mbatch % 1000 == 0:
            print('lr-batch-epoch:', opt.lr, param.nbatch, param.epoch)

        if mbatch >= 0 and mbatch % args.verbose == 0:
            acc_list = ver_test(mbatch)
            print(acc_list)
            save_step[0] += 1
            msave = save_step[0]
            do_save = False
            if len(acc_list) > 0:
                lfw_score = acc_list[0]

                # if lfw_score > highest_acc[0]:
                # if lfw_score >= 0.50:
                #     do_save = True
                #     highest_acc[0] = lfw_score
                    # 修改验证集阈值，测试最佳阈值
                    # if lfw_score>=0.998:
                if acc_list[-1] >= highest_acc[-1]:
                    highest_acc[-1] = acc_list[-1]
                    # if lfw_score>=0.99: #LFW测试大于0.99时,保存模型
                    if lfw_score >= 0.90:  # LFW测试大于0.99时,保存模型
                        do_save = True
            if args.ckpt == 0:
                do_save = False
            elif args.ckpt > 1:
                do_save = True
            if do_save:
                print('saving', msave)
                arg, aux = model.get_params()
                mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)

            print('[%d]Accuracy-Highest: %1.5f' % (mbatch, highest_acc[-1]))
        if mbatch <= args.beta_freeze:
            _beta = args.beta
        else:
            move = max(0, mbatch - args.beta_freeze)
            _beta = max(args.beta_min, args.beta * math.pow(1 + args.gamma * move, -1.0 * args.power))
        # print('beta', _beta)  5
        os.environ['BETA'] = str(_beta)
        if args.max_steps > 0 and mbatch > args.max_steps:
            sys.exit(0)

    epoch_cb = None
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    print('data fit...........')
    model.fit(train_data=train_dataiter,
              begin_epoch=begin_epoch,
              num_epoch=end_epoch,
              eval_data=None,
              eval_metric=eval_metrics,
              kvstore='device',
              optimizer=opt,
              # optimizer_params = optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              batch_end_callback=_batch_callback,
              epoch_end_callback=epoch_cb)

def main():
    # time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
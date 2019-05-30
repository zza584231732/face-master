# coding:utf-8
import face_model
import argparse
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import time
import datetime
import shutil

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../my_model/densenet169-model,0021', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=None, type=int, help='gpu id')
parser.add_argument('--image_path', default='../dataset/test_images/', help='test image path')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=0.78, type=float, help='ver dist threshold')
parser.add_argument('--threshold2', default=4, type=int, help='threshold for those whose dist is above threshold1')
args = parser.parse_args()

face_path = '../gallery/face_gallery.npy'
label_path = '../gallery/face_labels.npy'

faces = np.load(face_path)
labels = np.load(label_path)

testresult = open('result', 'w')


def test_Faces(args):
    model = face_model.FaceModel(args)
    imgs = os.listdir(args.image_path)
    # 设置阈值，这两个阈值用来判定该人是否是库里的人
    a = zip([args.threshold], [args.threshold2])
    for k, v in a:
        for img in imgs:
            flag = False
            start_time = time.time()
            pic = cv2.imread(os.path.join(args.image_path, img))
            pic = model.get_input(pic)

            if pic is None:
                continue
            else:
                f1 = model.get_feature(pic)
                for i in range(faces.shape[0]):
                    cnt = 0
                    for j in range(faces.shape[1]):
                        dist = np.sqrt(np.sum(np.square(f1 - faces[i][j])))
                        # 如果与库中某人距离小于阈值1
                        if dist < k:
                            cnt += 1
                    # 如果与库中某个人的相似度大于阈值2，则证明是该人
                    if cnt >= v:
                        name = labels[i]
                        testresult.writelines(os.path.join(args.image_path, img) + ' is ' + name + '\n')
                        flag = True
                        print(name)
                if flag is False:
                    print("image:" + os.path.join(args.image_path, img) + 'is not in the gallery, refused!' + '\n')
            end_time = time.time()
            print('time cost:',end_time-start_time)
    testresult.close()


if __name__ == '__main__':
    test_Faces(args)

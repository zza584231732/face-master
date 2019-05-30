# coding:utf-8
import sys
import os
import random
import time
import itertools
import pdb

src = '/home/zhang/tm/insightface_for_face_recognition-master/dataset/train_extface/extface_align_test/'
dst = open('/home/zhang/tm/insightface_for_face_recognition-master/dataset/test_pair.txt', 'a')

cnt = 0
same_list = []
diff_list = []
list1 = []
list2 = []
same_pirs = []
folders_1 = os.listdir(src)

# 产生相同的图像对
i=0
for folder in folders_1:

    sublist = []
    imgs = os.listdir(os.path.join(src, folder))
    for img in imgs:
        img_root_path = os.path.join(src, folder, img)
        sublist.append(img_root_path)
        list1.append(img_root_path)
    # 组合
    for item in itertools.combinations(sublist, 2):
        # for name in item:
        #     same_list.append(name)
        # for name in item:
        dst.writelines(item[0] + ' ' + item[1] + ' ' + '1' + '\n')
        i = i+1
        break
    # for j in range(0, len(list1), 2):
    #     dst.writelines(same_list[j] + ' ' + same_list[j + 1] + ' ' + '1' + '\n')
print(i)
list2 = list1.copy()
# 产生不同的图像对
diff = 0
# 如果不同的图像对远远小于相同的图像对，则继续重复产生，直到两者相差很小
while diff < i:
    random.seed(time.time() * 100000 % 10000)
    random.shuffle(list2)
    for p in range(0, len(list2)-1, 2):
        if list2[p] != list2[p + 1]:
            dst.writelines(list2[p] + ' ' + list2[p + 1] + ' ' + '-1' + '\n')
            diff += 1
        if diff < len(folders_1):
            continue
        else:
            break

from scipy.misc import imread,imresize,imsave
import os

path = '/home/zhang/tm/insightface_for_face_recognition-master/dataset/8631_align_train/'
out_path = '/home/zhang/tm/insightface_for_face_recognition-master/dataset/8631_112_align_train/'
img_lists = os.listdir(path)


for img_list in img_lists:
    imgpaths = os.path.join(path,img_list)
    out_imgpaths = os.path.join(out_path,img_list)
    if not os.path.exists(out_imgpaths):
        os.mkdir(out_imgpaths)
    img_names = os.listdir(imgpaths)
    for i in img_names:
        img_name = os.path.join(imgpaths,i)
        out_img_name = os.path.join(out_imgpaths,i)
        img = imread(img_name)
        img = imresize(img,(112,96))
        imsave(out_img_name,img)



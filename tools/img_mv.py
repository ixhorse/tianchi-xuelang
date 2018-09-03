"""
2018/7/17
move images
root/ants/xxx.png
root/ants/xxy.jpeg
root/ants/xxz.png
.
.
.
root/bees/123.jpg
root/bees/nsdf3.png
root/bees/asd932_.png
"""

import os, sys, shutil
import cv2

img_path = '../train/images'
ann_path = '../train/annotations'
output_path = '../images_test'
#datasets = ['train', 'val', 'train_all', 'test']
datasets = ['test_b']

if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.mkdir(output_path)

for dataset in datasets:
    data_path = os.path.join(output_path, dataset)
    os.mkdir(data_path)
    os.mkdir(os.path.join(data_path, '1'))
    os.mkdir(os.path.join(data_path, '0'))
    lst_file = '../train/' + dataset + '.lst'
    with open(lst_file, 'r') as f:
        db = [x.strip() for x in f.readlines()]
    
    if dataset == 'test_b':
        img_path = '../test_b'
    for obj in db:
        _, cls, img_name = obj.split('\t')
        img_name = img_name.split('/')[-1]
        print img_name
        # shutil.copy(os.path.join(img_path, img_name), os.path.join(data_path, cls, img_name))
        im = cv2.imread(os.path.join(img_path, img_name))
        im_new = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(data_path, cls, img_name), im_new)

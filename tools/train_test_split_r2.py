"""
2018/8/24
split train and test set
"""

import os, sys
import shutil
import random
import cv2

data_path = '../images'
train_all_path = '../train_all'
train_path = '../train'
val_path = '../val'

def mk_dirs():
    for dir_name in os.listdir(data_path):
        for set in [train_path, val_path, train_all_path]:
            cls_path = os.path.join(set, dir_name)
            if os.path.exists(cls_path):
                shutil.rmtree(cls_path)
            os.mkdir(cls_path)

def train_test_split():
    for cls in os.listdir(data_path):
        cls_path = os.path.join(data_path, cls)
        img_list = os.listdir(cls_path)
        # random
        idx = list(range(len(img_list)))
        random.shuffle(idx)
        p = int(0.7 * len(idx))
        # split
        for i in range(len(idx)):
            im= cv2.imread(os.path.join(cls_path, img_list[idx[i]]))
            im_new = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
            if i < p:
                cv2.imwrite(os.path.join(train_path, cls, img_list[idx[i]]), im_new)
            else:
                cv2.imwrite(os.path.join(val_path, cls, img_list[idx[i]]), im_new)
            cv2.imwrite(os.path.join(train_all_path, cls, img_list[idx[i]]), im_new)

        print('%s done.' % cls)

if __name__ == '__main__':
    mk_dirs()
    train_test_split()

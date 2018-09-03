# -*- coding: utf-8 -*-
"""
2018/8/26
"""

import sys, os
import cv2

data_root = '../'

def resize():
    test_path = os.path.join(data_root, 'images')
    dest_path = os.path.join(new_root, 'test')
    for file in os.listdir(test_path):
        img = cv2.imread(os.path.join(test_path, file))
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(dest_path, file), img)


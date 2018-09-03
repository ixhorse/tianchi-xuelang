"""
read rec file and show img
"""

import mxnet as mx
from PIL import Image
import cv2

record = mx.recordio.MXRecordIO('../train/train.rec', 'r')

while(1):
    header, img = mx.recordio.unpack_img(record.read())
    Image.fromarray(img).show()
    raw_input("Press Enter to continue...")
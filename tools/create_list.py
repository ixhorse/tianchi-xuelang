"""
2018/7/12
tianchi xuelang
create image list file for classification
"""

import os
import sys

def create_list(img_path, ann_path, list_file):
    images = os.listdir(img_path)
    annotations = [x[0:-4] for x in os.listdir(ann_path)]

    with open(list_file, 'w') as f:
        for idx in range(len(images)):
            img_name = images[idx]
            img_prefix = img_name[:-4]
            print img_prefix
            """
            if 'neg' in img_name:
                f.write('{}\t0\tcrop_images/{}\n'.format(idx, img_name))
            elif 'pos' in img_name:
                f.write('{}\t1\tcrop_images/{}\n'.format(idx, img_name))
            """
            if not img_prefix in annotations:
                f.write('{}\t0\t{}\n'.format(idx, img_name))
            else:
                f.write('{}\t1\t{}\n'.format(idx, img_name))

if __name__ == '__main__':
    img_path = '../test_b'
    ann_path = '../train/annotations'
    list_file = '../train/test_b.lst'
    create_list(img_path, ann_path, list_file)

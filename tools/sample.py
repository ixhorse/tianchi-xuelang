"""
2018/7/16
tianchi xuelang
crop to generate samples 800*800
"""

import os, sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from mxnet import nd
import shutil

def get_pos_neg(lst_file):
    with open(lst_file, 'r') as f:
        img_list = [x.strip() for x in f.readlines()]

    db = {}
    for img in img_list:
        data = img.split('\t')
        db[data[2].split('/')[-1]] = int(data[1])
    return db

def get_gtbox(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gtbox = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        box = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]
        gtbox.append(box)
    return gtbox

def box_overlap(gtboxes, box):
    for gtbox in gtboxes:
        x_min = max(gtbox[0], box[0])
        y_min = max(gtbox[1], box[1])
        x_max = min(gtbox[2], box[2])
        y_max = min(gtbox[3], box[3])
        if x_min < x_max and y_min < y_max:
            overlap = (x_max - x_min) * (y_max - y_min) /(gtbox[2] - gtbox[0]) * (gtbox[3] - gtbox[1])
            if overlap > 0.9:
                return True
            overlap = (x_max - x_min) * (y_max - y_min) /(box[2] - box[0]) * (box[3] - box[1])
            if overlap > 0.7:
                return True
    return False
    #     if box[0] < gtbox[0] and box[1] < gtbox[1] and box[2] > gtbox[2] and box[3] > gtbox[3]:
    #         return True
    #     elif gtbox[0] < box[0] and gtbox[1] < box[1] and gtbox[2] > box[2] and gtbox[3] >box[3]:
    #         return True
    # return False

def crop_and_save(img_name, img, box, cls, id, outdir):
    img_crop = img[box[1]:box[3], box[0]:box[2]]
    name_crop = img_name[:-4] + '_pos%d.jpg' % id if cls ==  1 \
                else img_name[:-4] + '_neg%d.jpg' % id
    cv2.imwrite(os.path.join(outdir, name_crop), img_crop)

def sample(lst_file, outdir):
    db = get_pos_neg(lst_file)
    it = 0
    dbsize = len(db)
    for img_name, img_cls in db.items():
        img = cv2.imread('../train/images/' + img_name)
        # (h, w, c)
        height, width, _ = img.shape
        
        # random sample
        pos = 0
        neg = 0
        # sample 10 pos and 10 neg for each image
        t = 0
        if img_cls == 0:
            pos = 10
        while pos < 10 or neg < 5:
            t = t + 1
            x_ct = np.random.randint(400, width-400)
            y_ct = np.random.randint(400, height-400)
            # xmin, ymin, xmax, ymax
            box = [x_ct-400, y_ct-400, x_ct+400, y_ct+400]

            if img_cls == 0:
                crop_and_save(img_name, img, box, 0, neg, outdir)
                neg += 1
            else:
                xml_name = img_name[:-4] + '.xml'
                gtbox = get_gtbox('../train/annotations/' + xml_name)
                # format : corner
                iou = nd.contrib.box_iou(nd.array(gtbox), nd.array(box).reshape(-1,4)).max().asscalar()
                # if iou > 0.9 and pos < 10:
                #     print('.......................')
                #     crop_and_save(img_name, img, box, 1, pos, outdir)
                #     pos += 1
                if box_overlap(gtbox, box) and pos < 10:
                    crop_and_save(img_name, img, box, 1, pos, outdir)
                    pos += 1
                elif iou == 0 and neg < 5:
                    crop_and_save(img_name, img, box, 0, neg, outdir)
                    neg += 1
            if t > 1000:
                break

        if it % 50 == 0:
            print 'image %d/%d' % (it, dbsize)
        print it
        it += 1

if __name__ == '__main__':
    output_dir = '../train/crop_images'
    lst_file = '../train/train_all.lst'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    sample(lst_file, output_dir)
"""
2018/7/13
split train and test set
"""

import os, sys
import numpy as np

def write_list(data, path):
    with open(path, 'w') as f:
        for i in data:
            f.write(i)

def train_test_split(dataset_lst, ratio):
    with open(dataset_lst, 'r') as f:
        data = f.readlines()

    size = len(data)
    train_size = int(size * ratio)
    np.random.shuffle(data)

    train_lst = '../train/train.lst'
    val_lst = '../train/val.lst'

    write_list(data[0:train_size], train_lst)
    print 'write train set.'
    write_list(data[train_size:], val_lst)
    print 'write val set'

if __name__ == '__main__':
    data_lst = '../train/train_all.lst'
    train_ratio = 0.8
    train_test_split(data_lst, train_ratio)
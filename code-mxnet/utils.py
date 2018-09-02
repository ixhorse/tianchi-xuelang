"""
2018/7/14
common functions
"""

import os, sys

import mxnet as mx
from mxnet import gluon, image, init, nd
import gluoncv
from pandas import DataFrame

output_dir = '../output'

def save_model(net, epoch):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    filename = '%s-%04d.params' % (net.name, epoch)
    net.save_parameters(os.path.join(output_dir, filename))

def load_model(net, epoch):
    filename = '%s-%04d.params' % (net.name, epoch)
    param_path = os.path.join(output_dir, filename)
    assert os.path.exists(param_path), \
    "params file not exist."

    net.load_parameters(param_path)

def write_result(preds, test_lst):
    with open(test_lst, 'r') as f:
        img_names = [x.strip().split('\t')[-1] for x in f.readlines()]
    preds = preds[:len(img_names)]
    data = {'filename':img_names, 'probability':preds}
    pd = DataFrame(data, columns=['filename', 'probability'])
    pd.to_csv('../output/results.csv', index=False)
        
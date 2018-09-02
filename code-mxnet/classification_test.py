"""
2018/7/14
tianchi xuelang
test set
"""

import mxnet as mx
import numpy as np
import os, time, shutil, sys

sys.path.append('/home/mcc/codes/gluon-cv')

from PIL import Image
from sklearn import metrics

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import viz
import gluoncv
from utils import *

per_device_batch_size = 32

num_gpus = 1
num_workers = 8
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = per_device_batch_size * max(num_gpus, 1)

# data prepare
mean_rgb = [123.68, 116.779, 103.939]
std_rgb = [58.393, 57.12, 57.375]
data_root = '/home/mcc/data/xuelang'

test_data = mx.io.ImageRecordIter(
    path_imgrec         = os.path.join(data_root, 'train', 'test.rec'),
    path_imgidx         = os.path.join(data_root, 'train', 'test.idx'),
    preprocess_threads  = num_workers,
    shuffle             = False,
    batch_size          = batch_size,

    resize              = 448,
    data_shape          = (3, 448, 448),
    mean_r              = mean_rgb[0],
    mean_g              = mean_rgb[1],
    mean_b              = mean_rgb[2],
    std_r               = std_rgb[0],
    std_g               = std_rgb[1],
    std_b               = std_rgb[2],
)

# model
model_name = 'resnet50_v2'
net = gluoncv.model_zoo.get_model(model_name, pretrained=False)
with net.name_scope():
    net.output = nn.Dense(2)

test_epoch = 40
load_model(net, test_epoch)
net.collect_params().reset_ctx(ctx)
net.hybridize()

tic = time.time()

all_pred = []
for i, batch in enumerate(test_data):
    # img = batch[0][0] 
    data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
    preds = [net(X).softmax() for X in data]
    all_pred.append(preds[0][:,1].reshape(-1).asnumpy())

all_pred = np.hstack(all_pred)

print('[Test] time cost: %f'%(time.time()-tic))

# write result
# make sure test data is not shuffled
all_pred = [round(x-1e-4, 4) for x in all_pred]
write_result(all_pred, os.path.join(data_root, 'train', 'test.lst'))
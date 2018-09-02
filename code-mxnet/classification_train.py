"""
2018/7/12
tianchi xuelang
classification
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
# Learning rate decay factor
lr_decay = 0.1
# Epochs where learning rate decays
lr_decay_epoch = [30, 100, np.inf]

# Nesterov accelerated gradient descent
optimizer = 'nag'
# Set parameters
optimizer_params = {'wd': 0.001, 'momentum': 0.9}
# 'learning_rate':0.001, 

num_gpus = 1
num_workers = 8
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = per_device_batch_size * max(num_gpus, 1)

# data prepare
jitter_param = 0.2
lighting_param = 0.1
mean_rgb = [123.68, 116.779, 103.939]
std_rgb = [58.393, 57.12, 57.375]
data_root = '/home/mcc/data/xuelang'

train_data = mx.io.ImageRecordIter(
    path_imgrec         = os.path.join(data_root, 'train', 'crop.rec'),
    path_imgidx         = os.path.join(data_root, 'train', 'crop.idx'),
    preprocess_threads  = num_workers,
    shuffle             = True,
    batch_size          = batch_size,

    data_shape          = (3, 448, 448),
    mean_r              = mean_rgb[0],
    mean_g              = mean_rgb[1],
    mean_b              = mean_rgb[2],
    std_r               = std_rgb[0],
    std_g               = std_rgb[1],
    std_b               = std_rgb[2],
    rand_mirror         = True,
    random_resized_crop = True,
    max_aspect_ratio    = 4. / 3.,
    min_aspect_ratio    = 3. / 4.,
    max_random_area     = 1,
    min_random_area     = 0.08,
    brightness          = jitter_param,
    saturation          = jitter_param,
    contrast            = jitter_param,
    pca_noise           = lighting_param,
)

val_data = mx.io.ImageRecordIter(
    path_imgrec         = os.path.join(data_root, 'train', 'val.rec'),
    path_imgidx         = os.path.join(data_root, 'train', 'val.idx'),
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

## model prepare
model_name = 'vgg19'
net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
with net.name_scope():
    net.output = nn.Dense(2)
# fix layers
# for i in range(0, 6):
#     net.features.__getitem__(i).collect_params().setattr('grad_req', 'null')
#     print net.features._children.values()
net.output.initialize(init.Xavier(), ctx = ctx)
net.collect_params().reset_ctx(ctx)
net.hybridize()

## trainer
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
acc_top1 = mx.metric.Accuracy()
# train_history = TrainingHistory(['training-top1-err', 'training-top5-err'])
L = gluon.loss.SoftmaxCrossEntropyLoss()

def test(ctx, val_data):
    all_label = []
    all_pred = []
    val_data.reset()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        preds = [net(X).softmax() for X in data]
        all_pred.append(preds[0][:,1].reshape(-1).asnumpy())
        all_label.append(label[0].asnumpy())
    
    all_label = np.int32(np.hstack(all_label))
    all_pred = np.hstack(all_pred)
    fpr, tpr, _ = metrics.roc_curve(all_label, all_pred, pos_label=1)
    return metrics.auc(fpr, tpr)

## train
epochs = 40
lr_decay_count = 0
log_interval = 50

for epoch in range(epochs):
    tic = time.time()
    btic = time.time()
    acc_top1.reset()

    if epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

    train_data.reset()
    for i, batch in enumerate(train_data):
        # img = batch[0][0]
        # break
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        with ag.record():
            outputs = [net(X) for X in data]
            loss = [L(yhat, y.astype('float32')) for yhat, y in zip(outputs, label)]
        ag.backward(loss)
        trainer.step(batch_size)
        acc_top1.update(label, outputs)
        if log_interval and not (i + 1) % log_interval:
            _, top1 = acc_top1.get()
            err_top1 = 1-top1
            print('Epoch[%d] Batch [%d]     Speed: %f samples/sec   top1-err=%f   loss=%f'%(
                      epoch, i, batch_size*log_interval/(time.time()-btic), err_top1, sum([l.mean().asscalar() for l in loss]) / len(loss)))
            btic = time.time()
    # break
    _, top1 = acc_top1.get()
    err_top1= 1-top1

    auc = test(ctx, val_data)
    # train_history.update([err_top1, err_top5, err_top1_val, err_top5_val])

    print('[Epoch %d] training: err-top1=%f'%(epoch, err_top1))
    print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
    print('[Epoch %d] validation: auc=%f'%(epoch, auc))
## save
# save_model(net, epochs)

# from matplotlib import pyplot as plt
# # Image.fromarray(np.uint8(img*256)).show()
# img = gluon.data.vision.datasets.ImageRecordDataset(val_rec)[100][0]
# print gluon.data.vision.datasets.ImageRecordDataset(val_rec)[100][1]
# viz.plot_image(img, reverse_rgb=False)
# viz.plot_image(img, reverse_rgb=True)
# plt.show()
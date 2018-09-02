#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test bilinear CNN.
2018/7/25
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from utils import test_model, submit_csv
import pickle


class BCNN(torch.nn.Module):
    """B-CNN for CUB200.

    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.

    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 200.
    """
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg19_bn(pretrained=False).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, 11)

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.autograd.Variable of shape N*3*448*448.

        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)
        X = X.view(N, 512, 28**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28**2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sign(X)*torch.sqrt(torch.abs(X)+1e-12)
        # X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 11)
        return X

data_transform = transforms.Compose([
        transforms.Resize(448),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# data_dir = '/home/mcc/data/xuelang'
data_dir = '../data/512'
image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test_r2'), data_transform)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=32,
                                             shuffle=False, num_workers=12)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = BCNN()
model_ft.load_state_dict(torch.load('../output/bcnn_step2_20180830_0953.pth'))
model_ft = model_ft.to(device)

all_prob = test_model(model_ft, dataloader, device)

img_names = sorted(os.listdir(os.path.join(data_dir, 'test_r2', 'norm')))
# submit_csv('bcnn', img_names, all_prob)

with open('../output/prob_bcnn.pkl', 'wb') as f:
    pickle.dump(all_prob, f)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune all layers for bilinear CNN.

modified from:https://github.com/HaoMood/bilinear-cnn.git

"""

import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from utils import train_model, save_model


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
        self.features = torchvision.models.vgg19(pretrained=False).features
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


# data
data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize(256),
        transforms.RandomResizedCrop(448, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(448),
        # transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# data_dir = '/home/mcc/data/xuelang'
data_dir = '../data/512'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train_all'), data_transforms['train'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
model_ft = BCNN()
model_ft.load_state_dict(torch.load('../output/bcnn_step1_20180830_1611.pth'))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001, momentum=0.9, weight_decay=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
num_epochs = 15

model_ft = train_model(model_ft, criterion, optimizer, exp_lr_scheduler,
                       num_epochs, dataloaders, dataset_sizes, device)

# save params
save_model(model_ft.state_dict(), 'bcnn_step2')
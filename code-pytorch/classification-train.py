# -*- coding: utf-8 -*-
"""
2018/7/17
train model
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os, sys
import copy
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np
import datetime
from utils import train_model, save_model


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# data_dir = '/home/mcc/data/xuelang'
data_dir = '../data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train_all'), data_transforms['train'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes
print(dataset_sizes, class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))

# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])

model_ft = models.resnet152(pretrained=True)

## freze
# it = 0
# cntr = 0
# for child in model_ft.children():
#     if cntr < it:
#         cntr += 1
#         for param in child.parameters():
#             param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 11)
# model_ft.classifier = nn.Sequential(
#             nn.Linear(512 * 14 * 14, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 11),
# )

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001, momentum=0.9, weight_decay=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
num_epochs = 10

model_ft = train_model(model_ft, criterion, optimizer, exp_lr_scheduler,
                       num_epochs, dataloaders, dataset_sizes, device)

# save params
save_model(model_ft.state_dict(), 'resnet152')
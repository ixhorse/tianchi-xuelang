"""
2018/7/17
test model
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn import metrics
from pandas import DataFrame
from PIL import Image
from utils import test_model, submit_csv
import pickle

# Data augmentation and normalization for training
# Just normalization for validation
data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# data_dir = '/home/mcc/data/xuelang'
data_dir = '../data'
image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test_r2'), data_transform)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=32,
                                             shuffle=False, num_workers=12)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet152(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 11)
# model_ft.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 2),
# )

model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load('../output/resnet152_20180830_1722.pth'))

all_prob = test_model(model_ft, dataloader, device)

img_names = sorted(os.listdir(os.path.join(data_dir, 'test_r2', 'norm')))
print(all_prob)
with open('../output/prob_resnet152.pkl', 'wb') as f:
        pickle.dump(all_prob, f)
# prob = [round(x-1e-4, 4) for x in prob]
# preds = [('defect_%d' % i) if i<10 else 'norm' for i in preds]
# data = {'filename':img_names, 'defect':preds, 'probability':preds}
# pd = DataFrame(data, columns=['filename', 'defect', 'probability'])
# pd.to_csv('../output/results_testb_vgg19_2.csv', index=False)
# submit_csv('resnet152', img_names, all_prob)
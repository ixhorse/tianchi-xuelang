# -*- coding: utf-8 -*-
"""
2018/8/26
"""

import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import datetime

def train_model(model, criterion, optimizer, scheduler, num_epochs,
                dataloaders, dataset_sizes, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_prob = []
            all_label = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if phase == 'train':
                    probability = nn.Softmax(dim=1)(outputs).detach().cpu().numpy()
                else:
                    probability = nn.Softmax(dim=1)(outputs).cpu().numpy()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_prob.append(probability)
                all_label.append(labels.cpu().numpy())

            # acc
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # auc
            all_label = np.int32(np.hstack(all_label))
            all_label = label_binarize(all_label, np.arange(11)) # one hot
            all_prob = np.vstack(all_prob)
            epoch_auc = metrics.roc_auc_score(all_label, all_prob, average='micro')

            print('{} Loss: {:.4f} Acc: {:.4f}, AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_auc))

            # deep copy the model
            if phase == 'val' and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
            
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Auc: {:4f}'.format(best_auc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def save_model(param, name):
    torch.save(param, '../output/'+name+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M') + '.pth')

def test_model(model, dataloader, device):
    since = time.time()
    model.eval()   # Set model to evaluate mode
    all_prob = []

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            probability = nn.Softmax(dim=1)(outputs).cpu().numpy()

        all_prob.append(probability)

    all_prob = np.vstack(all_prob)

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return all_prob

def submit_csv(model, filename, all_prob):
    csv_file = "../output/" + model + '_' +datetime.datetime.now().strftime('%Y%m%d_%H%M') + ".csv"
    class_name = [('defect_%d' % (j+1)) if j<10 else 'norm' for j in range(11)]
    with open(csv_file, 'w') as f:
        f.write('filename|defect,probability\n')
        for i in range(len(filename)):
            for j in range(11):
                f.write('%s|%s, %.4f\n' % (filename[i], class_name[j], all_prob[i,j]+1e-4))
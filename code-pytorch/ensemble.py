#!/usr/bin/env python
# -*- coding: utf-8 -*-\

"""
ensemble models
2018/7/31
"""

import pandas as pd
import numpy as np
import os
import pickle
from utils import submit_csv

output_dir = '../output'

weight = [0.6, 0.4]

def r1_ensemble():
    result_files = ['results_testb_vgg19.csv', 'results_testb_bcnn.csv', 'results_testb_vgg19_2.csv']
    results = []
    for result in result_files:
        results.append(pd.read_csv(os.path.join(output_dir, result)))

    img_names = []
    preds = []
    for i in range(len(results[0])):
        filename = results[0].loc[i]['filename']
        prob = 0
        for x in range(3):
            prob += weight[x] * results[x].loc[i]['probability']
        img_names.append(filename)
        preds.append(prob)

    preds = [round(x-1e-4, 4) for x in preds]
    data = {'filename':img_names, 'probability':preds}
    pd = pd.DataFrame(data, columns=['filename', 'probability'])
    pd.to_csv('../output/results_testb.csv', index=False)

def r2_ensemble():
    result_files = ['prob_bcnn.pkl', 'prob_resnet152.pkl']
    results = []
    for file in result_files:
        with open(os.path.join(output_dir, file), 'rb') as f:
            results.append(pickle.load(f))
    
    img_names = sorted(os.listdir(os.path.join('../data', 'test_r2', 'norm')))

    prob = np.zeros((len(img_names), 11))
    for i in range(len(img_names)):
        for j in range(11):
            tmp = 0
            for x in results:
                tmp += x[i][j]
            prob[i][j] = tmp / 2
    print(prob)
    submit_csv('ensemble', img_names, prob)

if __name__ == '__main__':
    r2_ensemble()
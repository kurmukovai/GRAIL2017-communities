'''
This script run with different arguments
computes roc auc score for overlapping (crisp)
approach with different number of clusters


usage example (for n_clusters=3, ok values are 3,4,5,6,7,8,9):

python grid_search_overlapping.py 3 



'''

import numpy as np
from model_evaluation import grid_search
import tqdm
import pickle
import pandas as pd
import sys

n_clusters = int(sys.argv[-1])

target = np.loadtxt('target')
subjects = np.loadtxt('subjects')
idx_ad_nc = np.loadtxt('idx_ad_nc')
idx_ad_nc = np.where(idx_ad_nc == 1, True, False)

thresh = ['05', '10', '15', '20', '25', '30', '35', '40']
path = 'precomputed_kernels/overlapping_kernels/{}_{}_{}'
colnames = ['kernel', '#clusters', 'threshold', 'kernel parameter', 'svc parameter', 'auc']
results = pd.DataFrame(columns = colnames)


for thr in tqdm.tqdm(thresh):        
    for method in tqdm.tqdm(['Omega', 'NMImax', 'NMIsum', 'lfkNMI']):

        with open(path.format(method, n_clusters, thr), 'rb') as f:
            kernel = pickle.load(f)

        train_auc_mean, train_auc_std, best_params, i, j = grid_search(kernel, target, subjects,
                                                               idx_ad_nc, kernel = 'exp', exact_matrix=True)

        temp = pd.DataFrame(data = [[method, n_clusters, thr, best_params['Kernel Parameter'],
                                    best_params['SVC Parameter'], train_auc_mean[i,j]]], columns=colnames)

        results = results.append(temp, ignore_index=True)

path2 = '/nmnt/media/home/anvar/conferences_code/MICCAI2017/reproducing_overlappingMICCAI/results_overlapping_AD_NC2.csv'
results.to_csv(path2)
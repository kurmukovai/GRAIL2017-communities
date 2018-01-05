# Computes 4 overlapping measures (crisp) for AD vs NC task

# Running this file (no parralelizm)
# takes about 8-10 hours 
# Consider start with using precomputed kernels,
# and if you want to be totally sure I did not fucked up
# recompute them using this file

import numpy as np
import os
from subprocess import Popen, PIPE
import re
import tqdm
import pickle

triu_grid = np.triu_indices(326, k=1)
i, j = triu_grid[0], triu_grid[1]
thresh = ['05', '10', '15', '20', '25', '30', '35', '40']

for n_clusters in tqdm.tqdm([2,3,4,5,6,7,8,9,10,11,12,13,14]):
    for thr in tqdm.tqdm(thresh):
       
        Omega = np.zeros((326, 326))
        NMImax = np.zeros((326, 326))
        lfkNMI = np.zeros((326, 326))
        NMIsum = np.zeros((326, 326))

        for u, v in tqdm.tqdm(zip(i, j)):

            cmd = 'Overlapping-NMI/onmi -o NMF_results/{}_{}_{} NMF_results/{}_{}_{}'.format(n_clusters,thr,u,
                                                                                              n_clusters,thr,v)

            proc = Popen(
               cmd,
               shell=True,
               stdout=PIPE, stderr=PIPE
            )
            proc.wait()
            res = proc.communicate()
            reg = re.findall('\D{1}(\d{1}\D{1}\d+)\D{1}', (str(res[0])))

            try:
                Omega[u, v] = float(reg[0])
            except:
                Omega[u, v] = 0
            try:
                NMImax[u, v] = float(reg[1])
            except:
                NMImax[u, v] = 0
            try:
                lfkNMI[u, v] = float(reg[2])
            except:
                lfkNMI[u, v] = 0
            try:
                NMIsum[u, v] = float(reg[3])
            except:
                NMIsum[u, v] = 0

        Omega += Omega.T
        NMIsum += NMIsum.T
        lfkNMI += lfkNMI.T
        NMImax += NMImax.T

        np.fill_diagonal(Omega, 1)
        np.fill_diagonal(NMIsum, 1)
        np.fill_diagonal(lfkNMI, 1)
        np.fill_diagonal(NMImax, 1)

        path = 'precomputed_kernels/overlapping_kernels/{}_{}_{}'

        with open(path.format('Omega', n_clusters, thr), 'wb') as f:
            pickle.dump(Omega,f)
        
        with open(path.format('NMImax', n_clusters, thr), 'wb') as f:
            pickle.dump(NMImax,f)

        with open(path.format('lfkNMI', n_clusters, thr), 'wb') as f:
            pickle.dump(lfkNMI,f)

        with open(path.format('NMIsum', n_clusters, thr), 'wb') as f:
            pickle.dump(NMIsum,f)

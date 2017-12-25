import numpy as np
from sklearn.metrics import adjusted_mutual_info_score


def compute_l2(M):
    n = M.shape[0]
    norm_l2 = np.zeros((n, n))
    for i in range(0, n):
        for j in range(i+1, n):
            dist = M[i] - M[j]
            l2 = np.sqrt(np.sum(np.power(dist, 2)))
            norm_l2[i, j] = l2
            norm_l2[j, i] = l2
    return norm_l2



def make_l1l2_kernel(pseudo_kernel, a):
        legit_kernel = np.exp(-a*(pseudo_kernel))
        return legit_kernel

def make_exp_kernel(pseudo_kernel, a):
        legit_kernel = np.exp(-a*(1-pseudo_kernel))
        return legit_kernel

def compute_AMI(all_lbl, mode='AMI'):

    if mode == 'AMI':
        metric_ami = np.diag(np.ones(all_lbl.shape[0]))
        rows, cols = np.triu_indices(metric_ami.shape[0], 1)

        for i, j in zip(rows, cols):
            sim_ami = adjusted_mutual_info_score(all_lbl[i], all_lbl[j])
            metric_ami[i, j] = sim_ami

        mat_ami = metric_ami + metric_ami.T
        np.fill_diagonal(mat_ami, 1)

        return mat_ami
    else:
        metric_ami = np.diag(np.ones(all_lbl.shape[0]))
        rows, cols = np.triu_indices(metric_ami.shape[0], 1)
        metric_ari = metric_ami.copy()

        for i, j in zip(rows, cols):
            sim_ari = adjusted_rand_score(all_lbl[i], all_lbl[j])
            metric_ari[i, j] = sim_ari
            sim_ami = adjusted_mutual_info_score(all_lbl[i], all_lbl[j])
            metric_ami[i, j] = sim_ami

        mat_ari = metric_ari + metric_ari.T
        np.fill_diagonal(mat_ari, 1)

        mat_ami = metric_ami + metric_ami.T
        np.fill_diagonal(mat_ami, 1)

        return mat_ari, mat_ami
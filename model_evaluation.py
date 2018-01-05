import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import time

params = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9] + list(range(1, 11))
penalties = np.array([0.01, 0.1, 1, 10, 50], dtype=float)


def grid_search(gram_matrix, target, subjects, idx,
                params=params, penalties=penalties,
                n_folds=10, n_repetitions=50,
                start_state=0, kernel = 'exp',
                exact_matrix=False):
    
    def return_unique(idx, target, subjects):
        
        

        y = target[idx]
        label = subjects[idx]

        encoder = dict(zip(np.unique(y), [0,1]))
        y = np.array([encoder[t] for t in y])

        df = pd.DataFrame(data = np.concatenate((y[:, np.newaxis],
                                                 label[:, np.newaxis]), axis=1),
                          columns=['target', 'subjects'])

        df.drop_duplicates('subjects', inplace=True)


    
        return y, df.target.values, label, df.subjects.values
    
    full_target_vector, target_unique, full_labels, names_unique = return_unique(idx, target, subjects)
    
    start = time.time()
    
    def exp_kernel(pseudo_kernel, a):
        legit_kernel = np.exp(-a*(1-pseudo_kernel))
        return legit_kernel

    def l1l2_kernel(pseudo_kernel, a):
        legit_kernel = np.exp(-a*(pseudo_kernel))
        return legit_kernel

    if kernel == 'exp':
        make_kernel = exp_kernel
    elif kernel == 'l1l2':
        make_kernel = l1l2_kernel
    
    
    train_auc_mean = np.zeros((len(params), len(penalties)))
    train_auc_std = np.zeros((len(params), len(penalties)))
    
    if not exact_matrix: # Можно передавать не полную матрицу 756x756 а уже поменьше с двумя AD Normal
        gram_matrix = gram_matrix[idx, :][:, idx]
    
        
    for kidx,a in enumerate(params):
        kernel = make_kernel(gram_matrix, a)
        for sidx,penalty in enumerate(penalties):
            auc = repeatSVM_labeled(kernel, target, subjects, idx,
                                    penalty = penalty)
            train_auc_mean[kidx, sidx] = np.mean(auc)
            train_auc_std[kidx, sidx] = np.std(auc)
    i, j = np.unravel_index(train_auc_mean.argmax(), train_auc_mean.shape)
    best_params = {'Kernel Parameter' : params[i], 'SVC Parameter' : penalties[j]}
    best_auc = train_auc_mean[i, j]
    
    print('Done {:.3f} sec'.format(time.time() - start))
    
    return train_auc_mean, train_auc_std, best_params, i, j


def repeatSVM_labeled(gram_matrix, target, subjects, idx,
                      n_folds=10, n_repetitions=50, start_state = 0,
                      penalty=1):    

    '''
    :inputs:
    
    :gram_matrix: square matrix n x n - pseudo kernel for SVM
    :target: n x 1 nd.array with correct labels for each observation
    :subjects: n x 1 nd.array - subject's ids
    :idx: True/False mask with True for those subjects that are considered in current task
    :n_folds: number of fold for cross validation
    :n_repetitions: number of different random states for StratifiedKFold
    :start_state: first random state  
    :penalty: - SVM regularization parameter
    
    :outputs:
        
    For each CV split we concatenate predictions on test fold into vector of size n x 1
    and compute ROC AUC score for it, this repeats n_repetitions times and all scores are stored in
    overall_roc_auc
    '''
    
    def return_unique(idx, target, subjects):
        
        

        y = target[idx]
        label = subjects[idx]

        encoder = dict(zip(np.unique(y), [0,1]))
        y = np.array([encoder[t] for t in y])

        df = pd.DataFrame(data = np.concatenate((y[:, np.newaxis],
                                                 label[:, np.newaxis]), axis=1),
                          columns=['target', 'subjects'])

        df.drop_duplicates('subjects', inplace=True)


    
        return y, df.target.values, label, df.subjects.values
    
    
    full_target_vector, target_unique, full_labels, names_unique = return_unique(idx, target, subjects)
    
    dummy = np.zeros((target_unique.shape[0], 2))
    
    X, y = gram_matrix, np.array(full_target_vector)
    labels = np.array(full_labels) 
    
    clf = SVC(C=penalty, kernel='precomputed', random_state = 0)
    overall_roc_auc=[]
   
    for rep in range(0, n_repetitions):
        
        CV = StratifiedKFold(n_folds, shuffle=True, random_state = start_state + rep)
        decision_predicted = np.zeros(y.shape[0])
        
        for train, test in CV.split(dummy, np.array(target_unique)):
            train_labels = np.array(names_unique)[train]
            test_labels = np.array(names_unique)[test]
            train_idx = np.in1d(labels, train_labels)
            test_idx = np.in1d(labels, test_labels)    
            
            clf.fit(X[train_idx][:, train_idx],y[train_idx])
            decision_output = clf.decision_function(X[test_idx][:, train_idx]) 
            decision_predicted[test_idx] = decision_output
      
        subject_prediction = np.zeros(np.array(target_unique).shape[0])
        
        for i in range(0, np.array(target_unique).shape[0]):
            lab = np.array(names_unique)[i]
            idx = np.where(labels==lab)[0]
            value = np.mean(decision_predicted[idx])
            subject_prediction[i] = value
                
        roc_auc_value = roc_auc_score(np.array(target_unique), subject_prediction)
        overall_roc_auc.append(roc_auc_value)
    
    return np.array(overall_roc_auc)

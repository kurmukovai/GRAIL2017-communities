import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix


def convert(data, size=68, mode = 'vec2mat', diag=None):
    '''
    Similar to scipy.spatial.distance.squareform but allows to 
    to convert square matrix with non-zero diagonal
    
    EXAMPLE :
    a = np.array([[1,2,3],[2,1,4],[3,4,1]])
    vec_a = convert(a, mode='mat2vec')
    print(vec_a)
    >>> array([2, 3, 4])
    convert(vec_a, size = 3, mode = 'vec2mat')
    >>> matrix([[1, 2, 3],
                [2, 1, 4],
                [3, 4, 1]], dtype=int64)
    '''

    if mode == 'mat2vec':
        
        mat = data.copy()
        rows, cols = np.triu_indices(data.shape[0], k=0)
        vec = mat[rows,cols]
        
        return vec

    elif mode == 'vec2mat':
        
        vec = data.copy()        
        rows, cols = np.triu_indices(size, k=0)
        mat = csr_matrix((vec, (rows, cols)), shape=(size, size)).todense()
        mat = mat + mat.T # symmetric matrix
        if diag is not None:
            np.fill_diagonal(mat, diag)
        else:
            np.fill_diagonal(mat, np.diag(mat)/2)
        
    return np.array(mat)

def load_ADNI(path):
    '''
    Simple script to import ADNI data set
    
    Данный набор данных содержит 807 снимков для 255 пациентов,
    каждому снимку поставлен в соответствие граф размера с 68 вершинами,
    метка класса (EMCI, Normal, AD, LMCI, SMC), а так же метка пациентов 
    (так как для каждого пациента есть несколько снимков,
    метки класса для одного пациента одинаковы для всех его снимков)
    
    IMPUT :
    
    path - this folder should contain folder "matrices"
           and 1 excel file ("ADNI2_Master_Subject_List.xls")
           
    OUTPUT : 
    
    data - numpy array of shape #subjects x #nodes x #nodes
    target - numpy array containing target variable
    data_copy - pandas dataframe containing 
                subject_id, 
                scan_id (multiple scans for some patients),
                adjacency matrices (data converted to vectors)
                target (diagnosis - AD, Normal, EMCI, LMCI, SMC)
                
    EXAMPLE : 
    
    path = 'notebooks to sort/connectomics/ADNI/Data'
    data, target, info = load_adni(path)
    
    
    TODO : 
    
    Add physical nodes position
    '''
    
    
    path_matrices = path + '/matrices/'
    path_subject_id = path + '/ADNI2_Master_Subject_List.xls'
    
    all_matrices = pd.DataFrame(columns = ['subject_id_file','subject_id','scan_id', 'matrix', 'target'])

    # import data
    for foldername in sorted(os.listdir(path_matrices)):
        for filename in sorted(os.listdir(path_matrices+foldername)):
            if 'NORM' not in filename:
                mat = np.genfromtxt(path_matrices+foldername+'/'+filename)
                subject_id_file = foldername
                subject_id = foldername[:-2]
                scan_id = foldername[-1:]

                # ADNI data have zeros on 3 and 38 row and column
                mat = np.delete(mat, [3,38], 1) 
                mat = np.delete(mat, [3,38], 0)

                subject_data = convert(mat, mode = 'mat2vec')
                single_subject = pd.DataFrame(data = [[subject_id_file, subject_id, scan_id, subject_data, np.nan]],
                                              columns = ['subject_id_file','subject_id','scan_id', 'matrix', 'target'])
                all_matrices = all_matrices.append(single_subject)

    all_matrices.index = all_matrices.subject_id_file
    subject_data = pd.read_excel(path_subject_id, sheetname = 'Subject List')
    subject_id_names = np.array(all_matrices['subject_id_file'])

    #importing target variables
    for name in subject_id_names:
        smth = subject_data.loc[subject_data['Subject ID'] == name[:-2]]['DX Group'].dropna()
        un_smth = np.unique(smth)
        try:
            val = un_smth[0].replace(' ', '')
            all_matrices.set_value(name, 'target', val)
        except:
            pass

    #drop objects without any target
    all_matrices.dropna(inplace = True)
    data_copy = all_matrices.copy()



    temp = data_copy['matrix']

    data_vectors = np.zeros((807, 2346))
    data = np.zeros((807, 68, 68))

    for idx, vec in enumerate(temp):
        data_vectors[idx] = vec
        data[idx] = convert(vec)

    target = all_matrices.target.values
    patients_ids = data_copy.subject_id.values

    print('ADNI data shape                   :', data.shape,
         '\nADNI target variable shape        :', target.shape,
         '\nADNI number of unique patients    :', data_copy.subject_id.unique().shape)
    return data, target, data_copy
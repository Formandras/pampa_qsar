import numpy as np
import torch

import sparsechem as sc
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

class ConfigObj:
    def __init__(self):
        self.fold_te                 = 0
            
        self.batch_ratio             = 0.5
        self.hidden_sizes            = [50]
        self.dropouts_trunk          = [0.7]
        self.last_non_linearity      = 'relu'
        self.weight_decay            =  0.0001
        self.lr                      =  0.001
        self.epochs                  =  20
            
        self.input_size_freq         = None
        self.dropouts_class          = []
        self.dropouts_reg            = []
        self.last_hidden_sizes       = None
        self.last_hidden_sizes_reg   = None
        self.last_hidden_sizes_class = None
        self.middle_non_linearity    = 'relu'
        self.enable_cat_fusion       = 0
        self.scaling_regularizer     = np.inf
        self.class_feature_size      = -1
        self.regression_feature_size = -1
        self.verbose                 = 0
        
        self.mixed_precision         = 0
        self.eval_frequency          = 1
        
        self.dev                     = 'cpu'
        # self.dev                     = 'cuda:0'

def prepare_dataloaders(input_repr, y_regr, folding, conf):
    input_repr = csr_matrix(input_repr)
    y_regr     = csr_matrix(y_regr)
    
    y_class    = csr_matrix((input_repr.shape[0], 0))
    y_censor   = csr_matrix(y_regr.shape)

    ## removing test data
    if conf.fold_te is not None and conf.fold_te >= 0:
        assert conf.fold_te != conf.fold_va, "fold_va and fold_te must not be equal."
        non_test_mask   = (folding != conf.fold_te)
        input_repr      = input_repr[non_test_mask]
        y_class         = y_class[non_test_mask]
        y_regr          = y_regr[non_test_mask]
        y_censor        = y_censor[non_test_mask]
        folding         = folding[non_test_mask]

    ## separate train, val
    tr_mask  = folding != conf.fold_va
    va_mask  = folding == conf.fold_va
    ##

    y_class_tr  = y_class[tr_mask]
    y_class_va  = y_class[va_mask]
    y_regr_tr   = y_regr[tr_mask]
    y_regr_va   = y_regr[va_mask]
    y_censor_tr = y_censor[tr_mask]
    y_censor_va = y_censor[va_mask]

    # TODO implement optional PCA here
    if len(conf.PCA_components) > 0:
        pca = PCA(n_components=max(conf.PCA_components)+1)
        # print(y_regr_tr.shape)
        # print(y_regr_tr.toarray().shape)
        pca.fit(y_regr_tr.toarray())    
        y_regr_tr   = csr_matrix(pca.transform(y_regr_tr.toarray())[:, conf.PCA_components])
        y_regr_va   = csr_matrix(pca.transform(y_regr_va.toarray())[:, conf.PCA_components])
        y_censor_tr = y_censor_tr[:, conf.PCA_components] # all 0 matrix, but shape has to fit
        y_censor_va = y_censor_va[:, conf.PCA_components] # all 0 matrix, but shape has to fit
        
        conf.measurement_cols = [f"PCA_{x}" for x in conf.PCA_components]
    
    batch_size  = int(np.ceil(conf.batch_ratio * tr_mask.sum()))

    dataset_tr = sc.ClassRegrSparseDataset(x=input_repr[tr_mask], y_class=y_class_tr, y_regr=y_regr_tr, y_censor=y_censor_tr)
    dataset_va = sc.ClassRegrSparseDataset(x=input_repr[va_mask], y_class=y_class_va, y_regr=y_regr_va, y_censor=y_censor_va)

    loader_tr = DataLoader(dataset_tr, batch_size=batch_size, num_workers = 1, pin_memory=True, collate_fn=dataset_tr.collate, shuffle=True)
    loader_va = DataLoader(dataset_va, batch_size=batch_size, num_workers = 1, pin_memory=True, collate_fn=dataset_va.collate, shuffle=False)

    # update conf with loader info
    conf.input_size        = dataset_tr.input_size
    conf.output_size       = dataset_tr.output_size
    conf.class_output_size = dataset_tr.class_output_size
    conf.regr_output_size  = dataset_tr.regr_output_size


    tasks_class = sc.load_task_weights(filename=None, y=y_class_tr, label="y_class")
    tasks_regr  = sc.load_task_weights(filename=None, y=y_regr_tr,  label="y_regr")

    tasks_class.training_weight = tasks_class.training_weight.to(conf.dev)
    tasks_regr.training_weight  = tasks_regr.training_weight.to(conf.dev)
    tasks_regr.censored_weight  = tasks_regr.censored_weight.to(conf.dev)
    
    tasks_class.aggregation_weight = np.ones(y_class_tr.shape[1]).astype(np.float64)
    tasks_regr.aggregation_weight  = np.ones(y_regr_tr.shape[1]).astype(np.float64)

    conf.tasks_class = tasks_class
    conf.tasks_regr = tasks_regr

    return loader_tr, loader_va



import os
import pandas as pd
import numpy as np
import torch
import time

import argparse
import json

import itertools
import sparsechem as sc
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.decomposition import PCA

from train_utils import load_datasets, own_compute_metrics_repr, own_compute_metrics_repr
from test_utils import ModelExperimentTest
from sparsechem_utils import prepare_dataloaders, ConfigObj

os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' #or CUBLAS_WORKSPACE_CONFIG=:16:8


parser = argparse.ArgumentParser(description="To run SKLearn models with config files.")
parser.add_argument('-p', '--config_path', type=str, help='The name of the JSON file containing the config parameter options for the experiment.')


# Parse the arguments
args = parser.parse_args()

with open(args.config_path, 'r') as json_file:
    config_options = json.load(json_file)
    
assert config_options['model_name'] == 'SparseChem', "This script can only be used with config files where 'model_name'=='SparseChem'."

print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
random_unique = int(np.random.random()*1000000000)

repr = config_options['representation']
train_measurement_cols = config_options['train_measurement_cols']
hidden_size = config_options['model_parameters']['hidden_size']
hidden_number = config_options['model_parameters']['hidden_number']
dropout = config_options['model_parameters']['dropout']
weight_decay =config_options['model_parameters']['weight_decay']
lr = config_options['model_parameters']['lr']

test_col = config_options['test_col']

all_repeats_best_pred_df, all_repeats_best_test_pred_df = None, None

for model_seed in config_options['seed']: 
    
    conf                = ConfigObj()
    conf.dev            = 'cpu'
    dev                 = torch.device(conf.dev)
    loss_class          = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss_regr           = sc.censored_mse_loss
    conf.epochs         = config_options['max_epoch']
    conf.PCA_components = [] # avoid doing PCA inside the dataloader
    PCA_components = [] if 'PCA_components' not in config_options.keys() else config_options['PCA_components']
                
    # here we need non-reproducable random IDs
    np.random.seed(random_unique)
    random_unique = int(np.random.random()*1000000000)
    
    # load both the train and test folds 
    exp = ModelExperimentTest(model = None, repr=repr, targets=train_measurement_cols, unique_random=random_unique)
    
    # prepare test matrices for SC without a loader
    X_coo       = coo_matrix(exp.X.astype(np.float32), exp.X.shape)
    X_coo       = torch.sparse_coo_tensor(np.vstack((X_coo.row, X_coo.col)), X_coo.data, exp.X.shape).to(dev)
    X_test_coo  = coo_matrix(exp.X_test.astype(np.float32), exp.X_test.shape)
    X_test_coo  = torch.sparse_coo_tensor(np.vstack((X_test_coo.row, X_test_coo.col)), X_test_coo.data, exp.X_test.shape).to(dev)
    
    
    # prepare PCA out od the CV loop for testing
    if len(PCA_components) > 0:
        pca = PCA(n_components=max(PCA_components)+1)
        # print(exp.Y.shape)
        pca.fit(exp.Y)
        
        exp.Y = pca.transform(exp.Y)[:, PCA_components]
        exp.Y_test  = pca.transform(exp.Y_test)[:, PCA_components]
        
        exp.measurement_cols = [f"PCA_{x}" for x in PCA_components]
    assert test_col in exp.measurement_cols, f'test_col="{test_col}" is not in exp.measurement_cols="{exp.measurement_cols}"'
        
    # configure hyperparameters
    conf.hidden_sizes   = [hidden_size] * hidden_number
    conf.dropouts_trunk = [dropout] * hidden_number
    conf.weight_decay   = weight_decay
    conf.lr             = lr    
    
    train_metric_df, valid_metric_df = None, None
    best_pred_per_task_per_fold, best_test_pred_per_task_per_fold = [], []
    for val_i in range(1,5):
        # here we need reproducable computations of the whole training cycle
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        print(val_i)
        conf.fold_va           = val_i
        conf.repr              = repr
        conf.measurement_cols  = exp.measurement_cols
        
        loader_tr, loader_va = prepare_dataloaders(
                                    input_repr  = exp.X,
                                    y_regr      = exp.Y,
                                    folding     = exp.folds,
                                    conf        = conf)
            
        net  = sc.SparseFFN(conf).to(dev)
        optimizer = torch.optim.Adam(net.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

        fold_train_metric_df, fold_valid_metric_df = None, None
        va_r2_max, va_r2_hist = None, []
        
        best_va_r2_per_task, best_pred_per_task, best_test_pred_per_task = None, None, None
        for epoch in range(conf.epochs):
            t0 = time.time()
            sc.train_class_regr(
                net, optimizer,
                loader          = loader_tr,
                loss_class      = loss_class,
                loss_regr       = loss_regr,
                dev             = dev,
                progress        = conf.verbose >= 2,
                weights_class   = conf.tasks_class.training_weight,
                weights_regr    = conf.tasks_regr.training_weight,
                censored_weight = conf.tasks_regr.censored_weight,
                args=conf
                )
            
            results_va = sc.evaluate_class_regr(net, loader_va, loss_class, loss_regr, 
                                                tasks_class=conf.tasks_class, tasks_regr=conf.tasks_regr, 
                                                dev=dev, progress = conf.verbose >= 2, normalize_inv=None, num_bins=10)
                            
            epoch_results_va_reg_r2 = results_va['regression']['rsquared']            
            with torch.no_grad():
                _, y_pred = net(X_coo)
                _, y_test_pred = net(X_test_coo)
            if best_va_r2_per_task is None:
                best_va_r2_per_task     = dict(epoch_results_va_reg_r2)
                best_pred_per_task      = y_pred
                best_test_pred_per_task = y_test_pred
            else:
                for i, task_r2 in epoch_results_va_reg_r2.items():
                    if best_va_r2_per_task[i] < task_r2:
                        best_va_r2_per_task[i] = task_r2
                        best_pred_per_task[:, i] = y_pred[:, i]
                        best_test_pred_per_task[:, i] = y_test_pred[:, i]
            
        # store best test predictions 
        best_pred_per_task_per_fold.append(best_pred_per_task)
        best_test_pred_per_task_per_fold.append(best_test_pred_per_task)
    
    ensemble_best_pred_per_task = torch.stack(best_pred_per_task_per_fold, dim=0).mean(dim=0)
    ensemble_best_test_pred_per_task = torch.stack(best_test_pred_per_task_per_fold, dim=0).mean(dim=0)
    
    ensemble_best_pred_df = pd.DataFrame(ensemble_best_pred_per_task, columns=conf.measurement_cols)
    ensemble_best_test_pred_df = pd.DataFrame(ensemble_best_test_pred_per_task, columns=conf.measurement_cols)
    
    ensemble_best_pred_df["qsar_id"] = exp.Y_ID.to_list()
    ensemble_best_test_pred_df["qsar_id"] = exp.Y_test_ID.to_list()
    ensemble_best_pred_df["model_seed"] = model_seed
    ensemble_best_test_pred_df["model_seed"] = model_seed
    
    if all_repeats_best_pred_df is None:
        all_repeats_best_pred_df = ensemble_best_pred_df
        all_repeats_best_test_pred_df = ensemble_best_test_pred_df
    else:
        all_repeats_best_pred_df = pd.concat([all_repeats_best_pred_df, ensemble_best_pred_df])
        all_repeats_best_test_pred_df = pd.concat([all_repeats_best_test_pred_df, ensemble_best_test_pred_df])
        
all_repeats_best_pred_df.reindex().to_csv(f'./modelling/control/best_preds_SC_{repr}_{test_col}.csv')
all_repeats_best_test_pred_df.reindex().to_csv(f'./modelling/control/best_test_preds_SC_{repr}_{test_col}.csv')

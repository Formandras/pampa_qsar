import os
import pandas as pd
import numpy as np
import torch
import time

import argparse
import json

import itertools
import sparsechem as sc
from scipy.sparse import csr_matrix

from train_utils import load_datasets, ModelExperiment, own_compute_metrics_repr
from sparsechem_utils import prepare_dataloaders, ConfigObj

os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' #or CUBLAS_WORKSPACE_CONFIG=:16:8


parser = argparse.ArgumentParser(description="To run SKLearn models with config files.")
parser.add_argument('-p', '--config_path', type=str, help='The name of the JSON file containing the config parameter options for the experiment.')
parser.add_argument('-c', '--comment', type=str, nargs='?', default='', help='Add a comment to the database concerning these experiments.')
parser.add_argument('-n', '--db_name', type=str, nargs='?', default='all_experiments', help='Add a comment to the database concerning these experiments.')
parser.add_argument('-t', '--db_table_name', type=str, nargs='?', default='experiments', help='Add a comment to the database concerning these experiments.')


# Parse the arguments
args = parser.parse_args()

with open(args.config_path, 'r') as json_file:
    config_options = json.load(json_file)
    
assert config_options['model_name'] == 'SparseChem', "This script can only be used with config files where 'model_name'=='SparseChem'."

print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
random_unique = int(np.random.random()*1000000000)
for repr, measurement_cols, model_seed, hidden_size, hidden_number, dropout, weight_decay, lr in itertools.product(
    config_options['representation'], config_options['measurement_cols'],  config_options['seed'], 
    config_options['model_parameters']['hidden_size'], config_options['model_parameters']['hidden_number'], 
    config_options['model_parameters']['dropout'], config_options['model_parameters']['weight_decay'], 
    config_options['model_parameters']['lr']): 
    
    conf                = ConfigObj()
    conf.dev            = 'cpu'
    dev                 = torch.device(conf.dev)
    loss_class          = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss_regr           = sc.censored_mse_loss
    conf.epochs         = config_options['max_epoch']
    conf.PCA_components = [] if 'PCA_components' not in config_options.keys() else config_options['PCA_components']
    
    # here we need non-reproducable random IDs
    np.random.seed(random_unique)
    random_unique = int(np.random.random()*1000000000)
    exp = ModelExperiment(model = None, repr=repr, targets=measurement_cols, unique_random=random_unique)
    train_metric_df, valid_metric_df = None, None

    conf.hidden_sizes   = [hidden_size] * hidden_number
    conf.dropouts_trunk = [dropout] * hidden_number
    conf.weight_decay   = weight_decay
    conf.lr             = lr

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
        conf.measurement_cols  = measurement_cols
        
        loader_tr, loader_va = prepare_dataloaders(
                                    input_repr  = exp.X,
                                    y_regr      = exp.Y,
                                    folding     = exp.folds,
                                    conf        = conf)
            
        net  = sc.SparseFFN(conf).to(dev)
        optimizer = torch.optim.Adam(net.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

        fold_train_metric_df, fold_valid_metric_df = None, None
        va_r2_max, va_r2_hist = None, []
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
                
            va_reg_agg_r2 = results_va['regression_agg']['rsquared']
            va_r2_hist.append(va_reg_agg_r2)
            
            # evalueate model and save result for all task
            results_tr = sc.evaluate_class_regr(net, loader_tr, loss_class, loss_regr, 
                                                tasks_class=conf.tasks_class, tasks_regr=conf.tasks_regr, 
                                                dev=dev, progress = conf.verbose >= 2, normalize_inv=None, num_bins=10)
            
            epoch_train_metric_df = results_tr['regression'].drop(['aggregation_weight'], axis=1)
            epoch_valid_metric_df = results_va['regression'].drop(['aggregation_weight'], axis=1)
            epoch_train_metric_df["target"] = conf.measurement_cols
            epoch_valid_metric_df["target"] = conf.measurement_cols
            epoch_train_metric_df["epoch"] = epoch
            epoch_valid_metric_df["epoch"] = epoch
            
            if fold_train_metric_df is None:
                fold_train_metric_df = epoch_train_metric_df
                fold_valid_metric_df = epoch_valid_metric_df
            else:
                fold_train_metric_df = pd.concat([fold_train_metric_df, epoch_train_metric_df])
                fold_valid_metric_df = pd.concat([fold_valid_metric_df, epoch_valid_metric_df])
        
        # select best model of this training for each task based on validation rsquared
        fold_train_metric_df.reset_index(inplace=True)
        fold_valid_metric_df.reset_index(inplace=True)
        
        max_ids = fold_valid_metric_df.groupby("target")["rsquared"].idxmax()
        fold_train_metric_df = fold_train_metric_df.loc[max_ids].sort_values('task').set_index('task').drop(['epoch'], axis=1)
        fold_valid_metric_df = fold_valid_metric_df.loc[max_ids].sort_values('task').set_index('task').drop(['epoch'], axis=1)
             
        if train_metric_df is None:
            train_metric_df = fold_train_metric_df
            valid_metric_df = fold_valid_metric_df
        else:
            train_metric_df = pd.concat([train_metric_df, fold_train_metric_df])
            valid_metric_df = pd.concat([valid_metric_df, fold_valid_metric_df])
            
    exp.train_metric_df = train_metric_df
    exp.valid_metric_df = valid_metric_df
        
    exp.model = 'SparseChem' # missuse: not a callable, but printable
    param_dict = {
        "hidden_sizes": conf.hidden_sizes, 
        "weight_decay": conf.weight_decay, 
        "dropouts_trunk": conf.dropouts_trunk, 
        "lr": conf.lr, 
        "seed": model_seed
    }
    if conf.measurement_cols != measurement_cols:
        param_dict["orig_measurement_cols"] = measurement_cols
        
    print(param_dict)
    
    exp.save_results(parameters=param_dict, comment=args.comment, save_fold_results=True, database_name=args.db_name, table_name=args.db_table_name)  
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

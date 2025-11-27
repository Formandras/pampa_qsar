import os
from datetime import datetime
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

import sparsechem as sc
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

def load_datasets(fp_repr='ecfp', measurement_cols=None, fp_csr=False, drop_nan_before_measurement_selection=False):
    pampa_qsar_home_lib = Path(__file__).absolute().parents[1]
    data_path = os.path.join(pampa_qsar_home_lib, 'data')
    
    # Load fingerprints
    assert fp_repr in ['ecfp', 'cddd', 'rdkit', 'percepta', 'molBERT'], "repr must be in ['ecfp', 'cddd', 'rdkit', 'percepta', 'molBERT']" 
    fingerprints = np.load(os.path.join(data_path, 'features', f'{fp_repr}.npy'), allow_pickle=True)[()]

    # Load meaurements
    target_df = pd.read_csv(os.path.join(data_path, 'all_plate_desalted_smiles_measurements_folds.csv'))
    folds = target_df['fold'].to_numpy()
    ## Select measurement cols to use
    if measurement_cols is None:
        measurement_cols = [col for col in target_df.columns if 'LogPe' in col]
    
    target_df = target_df[['qsar_id', 'fold',
                           'L_LogPe', 'PS_LogPe', 'DOD_LogPe', 'PC_LogPe', 'H_LogPe', 'BBB_LogPe',
                           'L_Pe',    'PS_Pe',    'DOD_Pe',    'PC_Pe',    'H_Pe',    'BBB_Pe', 
                           'L_MR',    'PS_MR',    'DOD_MR',    'PC_MR',    'H_MR',    'BBB_MR']]
    
    if not drop_nan_before_measurement_selection:
        target_df = target_df[['qsar_id', 'fold']+measurement_cols]
    
    nan_mask = target_df.isna().any(axis=1)
    target_df = target_df[~nan_mask]
    fingerprints = fingerprints[~nan_mask]
    folds = folds[~nan_mask]
    
    if drop_nan_before_measurement_selection:
        target_df = target_df[['qsar_id', 'fold']+measurement_cols]
        
    if fp_csr:
        fingerprints = csr_matrix(fingerprints)
    else:
        fingerprints = np.array(fingerprints)
        
    return fingerprints, target_df, folds

def own_compute_metrics_repr(Y_true, Y_pred, col_names=None):
    data = {
        "yr_col":  [],
        "yr_data": [],
        "yr_hat":  [],
    }
    num_tasks = Y_true.shape[1]
    if col_names is None:
        col_names = list(range(num_tasks))
    
    for col_id in range(num_tasks):
        if len(Y_pred.shape) == 1:
            Y_col_pred = Y_pred[:]            
        else:
            Y_col_pred = Y_pred[:, col_id]
            
        Y_col = Y_true[:, col_id]
        data["yr_col"].extend([col_id]*len(Y_col))
        data["yr_data"].extend(Y_col)
        data["yr_hat"].extend(Y_col_pred)
    
    data["yr_col"]  = np.asarray(data["yr_col"])
    data["yr_data"] = np.asarray(data["yr_data"])
    data["yr_hat"]  = np.asarray(data["yr_hat"])
    
    regr_metrics_df = sc.compute_metrics_regr(data["yr_col"], data["yr_data"], data["yr_hat"], num_tasks)
    regr_metrics_df["target"] = col_names
    return regr_metrics_df


class ModelExperimentBase:
    def __init__(self, model, repr, targets, PCA_components=[], unique_random=None, ravel=False) -> None:
        assert repr in ['ecfp', 'cddd', 'rdkit', 'percepta', 'molBERT'], "choose repr from ['ecfp', 'cddd', 'rdkit', 'percepta', 'molBERT']"
        target_ok_mask = np.in1d(targets, ['L_LogPe', 'PS_LogPe', 'DOD_LogPe', 'PC_LogPe', 'H_LogPe', 'BBB_LogPe', 'L_Pe', 'PS_Pe', 'DOD_Pe', 'PC_Pe', 'H_Pe', 'BBB_Pe', 'L_MR', 'PS_MR', 'DOD_MR', 'PC_MR', 'H_MR', 'BBB_MR'])
        assert target_ok_mask.all(), f"targets {np.array(targets)[np.invert(target_ok_mask)]} not found"
        
        self.model = model
        self.repr = repr
        self.measurement_cols = targets
        self.PCA_components = PCA_components
        self.ravel = ravel
        
        if unique_random is None:
            self.unique_random = int(np.random.random()*1000000000)
        else:
            self.unique_random = unique_random
        
        if repr == 'ecfp':
            fingerprints, target_df, folds = load_datasets(fp_repr=repr, measurement_cols=self.measurement_cols, fp_csr=True)
            fingerprints = sc.fold_transform_inputs(fingerprints, folding_size=2000)
            fingerprints = fingerprints.toarray()
        else:
            fingerprints, target_df, folds = load_datasets(fp_repr=repr, measurement_cols=self.measurement_cols, fp_csr=False)    
                
        # Split folds
        self.X = fingerprints.copy()
        self.Y = target_df[self.measurement_cols].to_numpy()
        self.Y_ID = target_df['qsar_id']
        self.folds = folds.copy()
    
    def collect_results_df(self, keep_fold_results=False):
        train_metric_mean_df = self.train_metric_df.groupby("target").mean().reset_index()
        valid_metric_mean_df = self.valid_metric_df.groupby("target").mean().reset_index()

        all_metric_mean_df = train_metric_mean_df.merge(valid_metric_mean_df, on='target', suffixes=('_train', '_valid'))
        
        if keep_fold_results:
            # collect train CV values in one row for each target
            train_metric_cv_df = self.train_metric_df.set_index(['target', self.train_metric_df.groupby('target').cumcount() + 1]).unstack()
            train_metric_cv_df.columns = ['_'.join(map(str, col)) for col in train_metric_cv_df.columns]
            train_metric_cv_df = train_metric_cv_df.reset_index()
            # collect valid CV values in one row for each target
            valid_metric_cv_df = self.valid_metric_df.set_index(['target', self.valid_metric_df.groupby('target').cumcount() + 1]).unstack()
            valid_metric_cv_df.columns = ['_'.join(map(str, col)) for col in valid_metric_cv_df.columns]
            valid_metric_cv_df = valid_metric_cv_df.reset_index()

            all_metric_cv_df = train_metric_cv_df.merge(valid_metric_cv_df, on='target', suffixes=('_train', '_valid'))
            
            all_metric_mean_df = all_metric_mean_df.merge(all_metric_cv_df, on='target')
        
        return all_metric_mean_df

class ModelExperiment(ModelExperimentBase):
    def __init__(self, model, repr, targets, PCA_components=[], unique_random=None, ravel=False) -> None:
        super().__init__(model, repr, targets, PCA_components, unique_random, ravel)
        
        # Split folds
        self.X = self.X[self.folds!=0]
        self.Y = self.Y[self.folds!=0]
        self.folds = self.folds[self.folds!=0]

    def cross_train_and_eval(self):
        self.train_metric_df, self.valid_metric_df = None, None

        for k in range(1, 5):
            X_train_fold, Y_train_fold = self.X[self.folds!=k], self.Y[self.folds!=k]
            X_valid_fold, Y_valid_fold = self.X[self.folds==k], self.Y[self.folds==k]
            
            if len(self.PCA_components) > 0:
                pca = PCA(n_components=max(self.PCA_components)+1)
                pca.fit(Y_train_fold)    
                
                Y_train_fold = pca.transform(Y_train_fold)[:, self.PCA_components]
                Y_valid_fold = pca.transform(Y_valid_fold)[:, self.PCA_components]
                
                
                self.measurement_cols = [f"PCA_{x}" for x in self.PCA_components]

            self.model.fit(X_train_fold, Y_train_fold if not self.ravel else Y_train_fold.ravel())
            Y_train_fold_pred = self.model.predict(X_train_fold)
            Y_valid_fold_pred = self.model.predict(X_valid_fold)
            # end fit and predict model

            train_metric_fold_df = own_compute_metrics_repr(Y_train_fold, Y_train_fold_pred, col_names=self.measurement_cols)
            valid_metric_fold_df = own_compute_metrics_repr(Y_valid_fold, Y_valid_fold_pred, col_names=self.measurement_cols)
            
            if self.train_metric_df is None:
                self.train_metric_df = train_metric_fold_df
                self.valid_metric_df = valid_metric_fold_df
            else:
                self.train_metric_df= pd.concat([self.train_metric_df, train_metric_fold_df])
                self.valid_metric_df= pd.concat([self.valid_metric_df, valid_metric_fold_df])
        
    def save_results(self, parameters, comment=None, save_fold_results=False, database_name='all_experiments', table_name='experiments'):
        
        save_df = self.collect_results_df(keep_fold_results=save_fold_results)
        
        save_df['unique_random']         = self.unique_random
        save_df['model']                 = str(self.model)
        save_df['input_representation']  = str(self.repr)
        save_df['train_on_columns']      = str(self.measurement_cols)
        save_df['parameters']            = str(parameters)
        save_df['comment']               = comment if comment is not None else ''
        save_df['timestamp']             = str(datetime.now())
        
        pampa_qsar_home_lib = Path(__file__).absolute().parents[1]
        data_path = os.path.join(pampa_qsar_home_lib, 'modelling')
        
        with sqlite3.connect(os.path.join(data_path, f'{database_name}.db')) as conn:
            save_df.to_sql(table_name, conn, if_exists='append', index=False)
    
        return save_df
    
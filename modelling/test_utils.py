import os
from datetime import datetime
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

import sparsechem as sc
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

from train_utils import load_datasets, own_compute_metrics_repr, ModelExperimentBase


class ModelExperimentTest(ModelExperimentBase):
    def __init__(self, model, repr, targets, PCA_components=[], unique_random=None, ravel=False) -> None:
        super().__init__(model, repr, targets, PCA_components, unique_random, ravel)
        
        # Split folds 
        # change this to 0 when real tests run
        self.X_test     = self.X[self.folds==0]
        self.Y_test     = self.Y[self.folds==0]
        self.Y_test_ID  = self.Y_ID[self.folds==0]
        self.folds_test = self.folds[self.folds==0]
        
        self.X          = self.X[self.folds>0]
        self.Y          = self.Y[self.folds>0]
        self.Y_ID       = self.Y_ID[self.folds>0]
        self.folds      = self.folds[self.folds>0]
                
    def cross_train_and_eval(self):
        self.train_metric_df, self.valid_metric_df = None, None
            
        if len(self.PCA_components) > 0:
            pca = PCA(n_components=max(self.PCA_components)+1)
            pca.fit(self.Y)    
            
            self.Y = pca.transform(self.Y)[:, self.PCA_components]
            self.Y_test = pca.transform(self.Y_test)[:, self.PCA_components]
            
            self.measurement_cols = [f"PCA_{x}" for x in self.PCA_components]

        test_pred_per_fold = []
        for k in range(1, 5):
            X_train_fold, Y_train_fold = self.X[self.folds!=k], self.Y[self.folds!=k]
            X_valid_fold, Y_valid_fold = self.X[self.folds==k], self.Y[self.folds==k]

            self.model.fit(X_train_fold, Y_train_fold if not self.ravel else Y_train_fold.ravel())
            Y_train_fold_pred = self.model.predict(X_train_fold)
            Y_valid_fold_pred = self.model.predict(X_valid_fold)
            Y_test_pred = self.model.predict(self.X_test)
            test_pred_per_fold.append(Y_test_pred)
            
            train_metric_fold_df = own_compute_metrics_repr(Y_train_fold, Y_train_fold_pred, col_names=self.measurement_cols)
            valid_metric_fold_df = own_compute_metrics_repr(Y_valid_fold, Y_valid_fold_pred, col_names=self.measurement_cols)
            
            if self.train_metric_df is None:
                self.train_metric_df = train_metric_fold_df
                self.valid_metric_df = valid_metric_fold_df
            else:
                self.train_metric_df= pd.concat([self.train_metric_df, train_metric_fold_df])
                self.valid_metric_df= pd.concat([self.valid_metric_df, valid_metric_fold_df])
        
        ensemble_test_pred = np.stack(test_pred_per_fold, axis=0).mean(0)
        self.test_metric_df = own_compute_metrics_repr(self.Y_test, ensemble_test_pred, col_names=self.measurement_cols)
        print(self.test_metric_df)
        self.test_metric_df = self.test_metric_df[self.test_metric_df['target']==self.test_col]
                
       
    def save_results(self, parameters, comment=None, save_fold_results=False, database_name='all_experiments', table_name='experiments'):
        
        save_df = self.collect_results_df(keep_fold_results=save_fold_results)
        self.test_metric_df.columns = [f'{col}_test' if col != 'target' else 'target' for col in self.test_metric_df.columns ]
        save_df = self.test_metric_df.merge(save_df, on='target', suffixes=('_test', None))
                
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

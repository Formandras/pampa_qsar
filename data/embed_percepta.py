import os
import pandas as pd
import numpy as np
import json

percepta_raw_df = pd.read_csv(os.path.join(".", "raw_data", "all_plate_percepta.csv")).drop(['*ID', 'compound_name', 'plate_number', 'orig_stock', 'desalted_SMILES', 'SMILES'], axis=1)
percepta_raw_df = percepta_raw_df.drop(['Dielectric Constant'], axis=1)
percepta_raw_df['qsar_id'] = percepta_raw_df['qsar_id'].apply(int)
percepta_raw_df = pd.concat([percepta_raw_df[['qsar_id', 'fold']], percepta_raw_df.drop(columns=['qsar_id', 'fold'])], axis=1)

# handle pKa 
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def choose_acid_base(pKa, choose):
    pKa_correct = 0
    if type(pKa) is float:
        if np.isnan(pKa):
            return 0.0
        pKa_correct = pKa
    elif type(pKa) is int:
        pKa_correct = float(pKa)
    else:
        pKa_list = [float(x) for x in pKa.split(';')]
        if choose == 'max':
            pKa_correct = max(pKa_list)
        elif choose == 'min':
            pKa_correct = min(pKa_list)
        else:
            assert False, 'only choose=["max", "min"] implemented'
    
    if choose == 'max':
        return sigmoid(pKa_correct-7.4)
    elif choose == 'min':
        return sigmoid(7.4-pKa_correct)

def choose_base_max(pKa):
    return choose_acid_base(pKa, 'max')
            
def choose_acid_min(pKa):
    return choose_acid_base(pKa, 'min')

orig_pKa_cols = ['pKa(Acid)|pKa', 'pKa(Acid)|Conf. limits', 'pKa(Acid)|AtomNo', 'pKa(Base)|pKa',  'pKa(Base)|Conf. limits', 'pKa(Base)|AtomNo']
percepta_raw_df['1st strongest acid pKa'] = percepta_raw_df['pKa(Acid)|pKa'].apply(choose_acid_min)
percepta_raw_df['1st strongest base pKa'] = percepta_raw_df['pKa(Base)|pKa'].apply(choose_base_max)
percepta_raw_df.drop(orig_pKa_cols, axis=1, inplace=True)

# get rid of Nan contaiing columns
columns_with_nan = percepta_raw_df.columns[percepta_raw_df.isna().any()].tolist()
percepta_raw_df.drop(columns_with_nan, axis=1, inplace=True)

# get rid of not sufficiently diverse colums
def filter_tiny_hist(column):
    values, counts = np.unique(column, return_counts=True)
    if len(values) == 1 or counts.sum()-counts.max()-counts[np.isnan(values)].sum() < 5:
        return True
    return False

drop_colums = [x for x in percepta_raw_df.columns if filter_tiny_hist(percepta_raw_df[x])]
print(drop_colums)
percepta_raw_df.drop(drop_colums, axis=1, inplace=True)

# normalize columns
no_test_mask = (percepta_raw_df['fold'] != 0)
feature_normal_stats_dict = {}
for i, feature_col in enumerate(list(percepta_raw_df.columns)[2:]):
    mean = percepta_raw_df[feature_col][no_test_mask].mean()
    std = percepta_raw_df[feature_col][no_test_mask].std()
    feature_normal_stats_dict[feature_col] = {
        "col_id": i,
        "mean": mean, 
        "std": std, 
    }
    percepta_raw_df[feature_col] = (percepta_raw_df[feature_col]-mean)/std

# save results
save_folder = os.path.join('.', 'data', 'features')
percepta_raw_df.to_csv(os.path.join(save_folder, 'percepta.csv'), index=False)
np.save(os.path.join(save_folder, 'percepta.npy'), percepta_raw_df.drop(['qsar_id', 'fold'], axis=1).to_numpy())

with open(os.path.join(save_folder, 'percepta_columns.json'), "w") as json_file:
    json.dump(feature_normal_stats_dict, json_file, indent=4)

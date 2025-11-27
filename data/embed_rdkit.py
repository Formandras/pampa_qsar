import os
import pandas as pd
import numpy as np
import json

from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles, Descriptors, MolSurf
from scipy.sparse import csr_matrix

plate_data = pd.read_csv(os.path.join('.', 'data', 'all_plate_desalted_smiles_measurements_folds.csv'))[['qsar_id', 'desalted_SMILES', 'fold']].sort_values(by=['qsar_id'])
assert (plate_data.index == plate_data['qsar_id']).all(), "qsar_id missing or not in correspondence with id"

rd_descriptors_dict = {
   "rdMolDescriptors": [
       "CalcChi0n", "CalcChi0v", "CalcChi1n", "CalcChi1v", "CalcChi2n", "CalcChi2v", "CalcChi3n", "CalcChi3v", "CalcChi4n", "CalcChi4v",
       "CalcExactMolWt", "CalcNumAtoms", "CalcFractionCSP3", "CalcHallKierAlpha",
       "CalcKappa1", "CalcKappa2", "CalcKappa3",
       "CalcNumAliphaticCarbocycles", "CalcNumAliphaticHeterocycles", "CalcNumAliphaticRings",  "CalcNumAmideBonds", "CalcNumAromaticCarbocycles", "CalcNumAromaticHeterocycles", "CalcNumAromaticRings",
       "CalcNumHBA", "CalcNumHBD", "CalcNumHeavyAtoms", "CalcNumHeteroatoms", "CalcNumHeterocycles", "CalcNumLipinskiHBA", "CalcNumLipinskiHBD", 
       "CalcNumRings", "CalcNumRotatableBonds", "CalcNumSaturatedCarbocycles", "CalcNumSaturatedHeterocycles", "CalcNumSaturatedRings", "CalcNumSpiroAtoms", "CalcNumBridgeheadAtoms",
       "CalcPhi", "CalcTPSA", "CalcLabuteASA","BCUT2D", "CalcCrippenDescriptors", 
        # check here: https://datagrok.ai/help/domains/chem/descriptors
        ],
   
   "Descriptors": [
       "ExactMolWt", "HeavyAtomMolWt", "MolWt", "NumValenceElectrons",
       "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", 
       "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge",
        # check here: https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html
       ],
   
   "MolSurf": [ 
       "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", 
       "SMR_VSA1", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SMR_VSA10",  
       "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12",
       "LabuteASA", "TPSA", "pyLabuteASA"
       # check: https://www.rdkit.org/docs/source/rdkit.Chem.MolSurf.html
   ]
}

rd_feature_rows = []
for _, row in plate_data.iterrows():
    qsar_id = row['qsar_id']
    desalted_smiles = row['desalted_SMILES']
    fold = row['fold']
    mol = MolFromSmiles(desalted_smiles)
        
    row_features_dict = {'qsar_id': qsar_id, 'fold': fold}

    for k, v_list in rd_descriptors_dict.items():
        module_obj = getattr(Chem, k)
        for v in v_list:
            attr_name = f'{k}_{v}'
            ret_attr = getattr(module_obj, v)(mol)
            try:
                for i, a in enumerate(ret_attr):
                    row_features_dict[f'{attr_name}_{i}'] = float(a)
            except TypeError:
                row_features_dict[attr_name] = float(ret_attr)
    
    rd_feature_rows.append(row_features_dict)
    
rd_features_df = pd.DataFrame.from_records(rd_feature_rows).sort_values(by=['qsar_id'])

def filter_tiny_hist(column):
    values, counts = np.unique(column, return_counts=True)
    if len(values) == 1 or counts.sum()-counts.max() < 5:
        return True
    return False
    
    
drop_colums = [x for x in rd_features_df.columns if filter_tiny_hist(rd_features_df[x])]
rd_features_df.drop(drop_colums, axis=1, inplace=True)

no_test_mask = (rd_features_df['fold'] != 0)
feature_normal_stats_dict = {}
for i, feature_col in enumerate(list(rd_features_df.columns)[2:]):
    # calculate values without test set
    mean = rd_features_df[feature_col][no_test_mask].mean()
    std = rd_features_df[feature_col][no_test_mask].std()
    feature_normal_stats_dict[feature_col] = {
        "col_id": i,
        "mean": mean, 
        "std": std, 
    }
    # aplly to all
    rd_features_df[feature_col] = (rd_features_df[feature_col]-mean)/std
    

save_folder = os.path.join('.', 'data', 'features')
rd_features_df.to_csv(os.path.join(save_folder, 'rdkit.csv'), index=False)
np.save(os.path.join(save_folder, 'rdkit.npy'), rd_features_df.drop(['qsar_id', 'fold'], axis=1).to_numpy())

with open(os.path.join(save_folder, 'rdkit_columns.json'), "w") as json_file:
    json.dump(feature_normal_stats_dict, json_file, indent=4)

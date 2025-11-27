import os
import sys
import inspect
import pandas as pd
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from MolBERT.molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer

path_to_checkpoint = os.path.join('.', 'MolBERT', 'molbert_100epochs', 'checkpoints', 'last.ckpt')
featurizer = MolBertFeaturizer(path_to_checkpoint)
print('model crated')

pampa_all_plates_df = pd.read_csv(os.path.join('.', 'data','all_plate_desalted_smiles_measurements_folds.csv'))[['qsar_id', 'compound_name', 'desalted_SMILES', 'fold']]
# print(pampa_all_plates_df)

pampa_smls = pampa_all_plates_df['desalted_SMILES'].tolist()
# print(pampa_smls)

pampa_emb, masks = featurizer.transform(pampa_smls)
pampa_emb = (pampa_emb - pampa_emb.mean(0)) / pampa_emb.std(0)
    
assert all(masks)
print(pampa_emb.shape)

# print(pampa_emb)
np.save(os.path.join('.', 'data', 'features', 'molBERT.npy'), pampa_emb)


bert_df = pd.DataFrame(data=pampa_emb, 
                        index=pampa_all_plates_df['qsar_id'], 
                        columns=[f'molBERT_{i}' for i in range(pampa_emb.shape[1])])

bert_df = pd.merge(pampa_all_plates_df[['qsar_id', 'fold']], bert_df, on='qsar_id')
bert_df.to_csv(os.path.join('.', 'data', 'features', 'molBERT.csv'), index=False)

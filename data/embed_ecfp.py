import os
import pandas as pd
import numpy as np

from rdkit.Chem import AllChem, MolFromSmiles
from scipy.sparse import csr_matrix

plate_data = pd.read_csv(os.path.join('.', 'data', 'all_plate_desalted_smiles_measurements_folds.csv'))[['qsar_id', 'desalted_SMILES', 'fold']].sort_values(by=['qsar_id'])
assert (plate_data.index == plate_data['qsar_id']).all(), "qsar_id missing or not in correspondence with id"

# prepare ECFP fingerprint matrix

raw_features = dict()
for _, row in plate_data.iterrows():
    qsar_id = row['qsar_id']
    desalted = row['desalted_SMILES']
    mol = MolFromSmiles(desalted)
    raw_features[qsar_id] = AllChem.GetMorganFingerprint(mol, 3).GetNonzeroElements()

ecfp_fold = 32000
fp2 =[(x,(np.array(list(raw_features[x].keys())), np.array(list(raw_features[x].values())))) for x in raw_features]
cmpd, ecfp   = zip(*fp2)
feat, counts = zip(*ecfp)
lens    = np.array([len(f) for f in feat])
indptr  = np.concatenate([[0], np.cumsum(lens)])
indices = np.concatenate(feat) % ecfp_fold
data = np.ones(indices.shape[0])
csr = csr_matrix((data, indices, indptr), shape=(len(feat), ecfp_fold))
csr.sum_duplicates()
csr.data[:] = 1.0

assert list(cmpd) == list(plate_data['qsar_id']), "order of ecfp fingrprints does not match order of qsar_ids in original DataFrame"

os.makedirs(os.path.join('.', 'data', 'features'), exist_ok=True)
np.save(os.path.join('.', 'data', 'features', 'ecfp.npy'), csr)

plate_data['ecfp_indices'] = [str(list(csr[i].indices)) for i in range(csr.shape[0])]
plate_data[["qsar_id", "fold", "ecfp_indices"]].to_csv(os.path.join('.', 'data', 'features', 'ecfp.csv'), index=False)
import os
import pandas as pd
import numpy as np
import json

from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles, rdFreeSASA
from scipy.sparse import csr_matrix

plate_data = pd.read_csv(os.path.join('.', 'data', 'all_plate_desalted_smiles_measurements_folds.csv'))[['qsar_id', 'desalted_SMILES', 'SMILES', 'compound_name', 'fold']].sort_values(by=['qsar_id'])
assert (plate_data.index == plate_data['qsar_id']).all(), "qsar_id missing or not in correspondence with id"

def compute_sasa(mol):
    """Compute Solvent Accessible Surface Area.
    """
    mol = Chem.AddHs(mol, addCoords=True)
    AllChem.EmbedMolecule(mol)
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [7,8,15,16]:
            atom.SetProp("SASAClassName", "Polar") # mark as polar
        elif atom.GetAtomicNum() == 1:
            if atom.GetBonds()[0].GetOtherAtom(atom).GetAtomicNum() in [7,8,15,16]:
                atom.SetProp("SASAClassName", "Polar") # mark as polar
        else:
            atom.SetProp("SASAClassName", "Apolar") # mark as apolar

    # Get Van der Waals radii (angstrom)
    ptable = Chem.GetPeriodicTable()
    radii = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]

    # Compute solvent accessible surface area
    sa = rdFreeSASA.CalcSASA(mol, radii, confIdx=-1, query=rdFreeSASA.MakeFreeSasaAPolarAtomQuery())
    
    return sa

def compute_vdwsa(mol):
    """Compute Van der Waals Surface Area.
    """
    mol = Chem.AddHs(mol)
    
    AllChem.EmbedMolecule(mol)
    
    radii = rdFreeSASA.classifyAtoms(mol)
    sa = rdFreeSASA.CalcSASA(mol, radii, confIdx=-1)
    return sa


rd_feature_rows = []
for _, row in plate_data.iterrows():
    qsar_id = row['qsar_id']
    desalted_smiles = row['desalted_SMILES']
    fold = row['fold']
    compound_name = row['compound_name']
    SMILES = row['SMILES']
    
    row_features_dict = {
        'qsar_id': qsar_id, 
        'compound_name': compound_name, 
        'SMILES': SMILES,
        'fold': fold}
    
    # try:    
    mol = MolFromSmiles(desalted_smiles)
    SASA_values = compute_sasa(mol)
    vdwsa_values = compute_vdwsa(mol)
    # except Exception as e:
    #     SASA_values = -1
    
    row_features_dict['SASA'] = SASA_values
    row_features_dict['vdwsa'] = vdwsa_values
    
    rd_feature_rows.append(row_features_dict)
    
rd_features_df = pd.DataFrame.from_records(rd_feature_rows).sort_values(by=['qsar_id'])

print(rd_features_df.head())


save_folder = os.path.join('.', 'data', 'features')
rd_features_df.to_csv(os.path.join(save_folder, 'SASA.csv'), index=False)
np.save(os.path.join(save_folder, 'SASA.npy'), rd_features_df.drop(['qsar_id', 'fold'], axis=1).to_numpy())


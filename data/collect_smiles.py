import os
import pandas as pd
from desalter import MyStandardizerDesalter

import numpy as np

plate_compound_names = pd.read_csv(os.path.join('.', 'data', 'all_plate_measurements_mean.csv'))#[['qsar_id', 'compound_name', 'plate_number']]
plate_compound_names['plate_number'] = plate_compound_names['plate_number'].apply(int) 
keep_columns = ['qsar_id', 'compound_name', 'plate_number', 'SMILES', 
                'L_LogPe', 'L_Pe', 'L_MR',
                'PS_LogPe', 'PS_Pe', 'PS_MR',
                'DOD_LogPe', 'DOD_Pe', 'DOD_MR',
                'PC_LogPe', 'PC_Pe', 'PC_MR',
                'H_LogPe', 'H_Pe', 'H_MR', 
                'BBB_LogPe', 'BBB_Pe', 'BBB_MR']

# ORIGINAL molecule bank (BTRG department)
BTRG_perceptra_in_path = os.path.join('.', 'raw_data', 'CELSA_Mcule_BTRG_egyesitett.xlsx')
BTRG_perceptra_df = pd.read_excel(BTRG_perceptra_in_path, sheet_name='BTRG_MolBank_Percepta frissites').drop(['*ID'], axis=1)
BTRG_perceptra_df.drop_duplicates(subset=['Name'], inplace=True)

BTRG_merged_df = plate_compound_names.merge(BTRG_perceptra_df, left_on='compound_name', right_on='Name').drop(['REF No', 'Name'], axis=1)
BTRG_merged_df = BTRG_merged_df[keep_columns]
BTRG_merged_df['orig_stock'] = "BTRG"

# molecules ordered from MCULE 
MCULE_perceptra_df = BTRG_perceptra_df[BTRG_perceptra_df.apply(lambda x: type(x['REF No'])== str and x['REF No'][0]=='M', axis=1)].copy()
MCULE_perceptra_df['own_mcule_id'] = [f'Mcule_{int(x[1:]):02d}' for x in MCULE_perceptra_df['REF No']]

MCULE_merged_df = plate_compound_names.merge(MCULE_perceptra_df, left_on='compound_name', right_on='own_mcule_id')
MCULE_merged_df['compound_name'] = MCULE_merged_df['Name']
MCULE_merged_df = MCULE_merged_df[keep_columns]
MCULE_merged_df['orig_stock'] = "MCULE"

# molecules selected from SOTE stock
SOTE_percepta_path = os.path.join('.', 'raw_data', 'Molekulabank SE GYTK GYKI.xlsx')
SOTE_percepta_df = pd.read_excel(SOTE_percepta_path).drop(['*ID', 'ID'], axis=1)

SOTE_percepta_df['name'] = SOTE_percepta_df['name'].apply(lambda x: f'{x}'.capitalize().strip())
plate_compound_names['capital_name'] = plate_compound_names['compound_name'].apply(lambda x: f'{x}'.capitalize().strip())

SOTE_merged_df = plate_compound_names.merge(SOTE_percepta_df, left_on='capital_name', right_on='name').drop(['name', 'iupac'], axis=1)
SOTE_merged_df = SOTE_merged_df[keep_columns]
SOTE_merged_df['orig_stock'] = "SOTE"

# concatenate the the sources and drop duplicates (keeping SOTE if also found in BTRG)
all_plate_percepta = pd.concat([BTRG_merged_df, MCULE_merged_df, SOTE_merged_df]) \
    .sort_values(by=['qsar_id', 'orig_stock']) \
    .drop_duplicates(subset='qsar_id', keep='last') \
    .reset_index(drop=True)


# check if any name is not mapped to a SMILES
for i in range(len(plate_compound_names)):
    if i not in all_plate_percepta['qsar_id'].to_numpy():
        print(i, plate_compound_names[plate_compound_names['qsar_id']==i].to_numpy())
        assert False
        
# desalt and standardize all molecules
myDesalter = MyStandardizerDesalter()
all_plate_percepta['desalted_SMILES'] = all_plate_percepta['SMILES'].apply(myDesalter.desalt_smiles)

np.random.seed(0)
all_plate_percepta['fold'] = np.random.permutation([i%5 for i in range(len(all_plate_percepta))]).tolist()

# save updated dataset
all_plate_percepta.to_csv(os.path.join('.', 'data', 'all_plate_desalted_smiles_measurements_folds.csv'), index=False)
all_plate_percepta[['qsar_id', 'compound_name', 'plate_number', 'orig_stock', 'desalted_SMILES']].to_csv(os.path.join('.', 'data', 'all_plate_desalted_smiles.csv'), index=False)

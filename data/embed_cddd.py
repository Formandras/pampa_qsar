import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from cddd.inference import InferenceModel


def main(unused_argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model_dir = os.path.join('.', 'cddd', 'default_model')

    infer_model = InferenceModel(model_dir, use_gpu=False, cpu_threads=5)
    print('model crated')
    
    pampa_all_plates_df = pd.read_csv(os.path.join('.', 'data','all_plate_desalted_smiles_measurements_folds.csv'))[['qsar_id', 'compound_name', 'desalted_SMILES', 'fold']]
    pampa_smls = pampa_all_plates_df['desalted_SMILES'].tolist()
    print("Extracting CDDD molecular desscriptors for all plates")
    pampa_emb = infer_model.seq_to_emb(pampa_smls)

    pampa_emb = (pampa_emb - pampa_emb.mean()) / pampa_emb.std()
    
    # print(pampa_emb)
    np.save(os.path.join('.', 'data', 'features', 'cddd.npy'), pampa_emb)
    
    cddd_df = pd.DataFrame(data=pampa_emb, 
                           index=pampa_all_plates_df['qsar_id'], 
                           columns=[f'cddd_{i}' for i in range(pampa_emb.shape[1])])

    cddd_df = pd.merge(pampa_all_plates_df[['qsar_id', 'fold']], cddd_df, on='qsar_id')
    cddd_df.to_csv(os.path.join('.', 'data', 'features', 'cddd.csv'), index=False)

    
    

if __name__ == "__main__":
    tf.app.run(main=main, argv=[sys.argv[0]])

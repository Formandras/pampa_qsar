import os
import json

conf_base = {
    "model_name": "SparseChem",
    "max_epoch": 200,
    "model_parameters": {
        "hidden_size"   : [5, 10, 15],
        "hidden_number" : [1],
        "dropout"       : [0.6, 0.5],
        "weight_decay"  : [0.01, 0.001],
        "lr"            : [0.1, 0.3]
    },
}
repr_opts = ["cddd"]
measurement_col_opts =  ["all_LogPe", "L_LogPe", "PS_LogPe", "DOD_LogPe", "PC_LogPe", "H_LogPe", "BBB_LogPe"]
seed_opt = [0, 1, 2, 3, 4]

for repr in repr_opts:
    for measurement_col in measurement_col_opts:
        for seed in seed_opt:
            conf = {
                "representation": [repr],
                "measurement_cols": [[measurement_col]] if measurement_col != "all_LogPe" else [["L_LogPe", "PS_LogPe", "DOD_LogPe", "PC_LogPe", "H_LogPe", "BBB_LogPe" ]],
                "seed" : [seed]
            }
            conf.update(conf_base)
            
            file_name = f"SparseChem_x_{repr}_y_{measurement_col}_L_1_s_{seed}.json"
            file_path = os.path.join("modelling", "config_files", "SparseChem_paralel", file_name)
            
            with open(file_path, 'w') as json_file:
                json.dump(conf, json_file, indent=4)

for repr in repr_opts:
    for PCA_col in [[0], [1], [2], [0,1,2]]:
        for seed in seed_opt:
            conf = {
                "representation": [repr],
                "measurement_cols": [["L_LogPe", "PS_LogPe", "DOD_LogPe", "PC_LogPe", "H_LogPe", "BBB_LogPe" ]],
                "seed" : [seed],
                "PCA_components": PCA_col,
            }
            conf.update(conf_base)
            
            file_name = f"SparseChem_PCA{''.join([str(c) for c in PCA_col])}_x_{repr}_L_1_s_{seed}.json"
            file_path = os.path.join("modelling", "config_files", "SparseChem_paralel", file_name)
            
            with open(file_path, 'w') as json_file:
                json.dump(conf, json_file, indent=4)
            

        
        
import os
import pandas as pd
import numpy as np

raw_measurements_path = os.path.join(".", "raw_data", "all_measurements")

measurement_files = [f for f in os.listdir(path=raw_measurements_path)]
measurement_files.sort()

measurement_df_list = []
for measurement_file in measurement_files:
    plate_num = int(measurement_file.split('_')[1][2:])
    measurement_file_path = os.path.join(raw_measurements_path, measurement_file)
    measurement_df = pd.read_excel(measurement_file_path)
    
    membrane_names = measurement_df.iloc[0].to_numpy()
    assay_names = measurement_df.iloc[1].to_numpy()
    
    # sort out columns names
    curr_membrane, curr_assay = 'compound', 'name'
    merged_col_names = []
    for i, (mem_val, ass_val) in enumerate(zip(membrane_names, assay_names)):
        if i > 24: break
        if type(mem_val) is str:
            curr_membrane = mem_val.split('(')[1].split(')')[0]
        if type(ass_val) is str:
            curr_assay = ass_val
        merged_col_names.append(f'{curr_membrane}_{curr_assay}')
    
    measurement_df.drop([0, 1], axis=0, inplace=True)
    measurement_df.drop(measurement_df.columns[[25, 26, 27, 28, 29]], axis=1, inplace=True)
    measurement_df.columns = merged_col_names
    
    drop_class_names = [class_col for class_col in merged_col_names if 'Class' in class_col]
    measurement_df.drop(drop_class_names, axis=1, inplace=True)
    
    # rename rows
    new_compound_names = measurement_df['compound_name'][[(i // 3) * 3 + 2 for i in range(measurement_df.shape[0])]]
    print(plate_num, len(new_compound_names)/3)
    measurement_df['compound_name'] = new_compound_names.to_numpy()
    
    # check values
    
    for col_name in measurement_df.columns:
        if '_LogPe' in col_name:
            pe_col = ''.join(col_name.split("Log"))
            
            underflow_mask = (measurement_df[col_name] == '-')
            measurement_df[pe_col]    = np.where(underflow_mask, 1e-7, measurement_df[pe_col])
            measurement_df[col_name] = np.where(underflow_mask, -7.0, measurement_df[col_name])
    
    measurement_df['plate_number'] = plate_num
    measurement_df[measurement_df == '-'] = np.nan
    
    measurement_df_list.append(measurement_df)
    
all_plate_measurement_df = pd.concat(measurement_df_list, axis=0, ignore_index=True)
all_plate_measurement_df.to_csv(os.path.join(".", "data", "all_plate_measurements.csv"))


all_plate_measurement_mean_df = all_plate_measurement_df.groupby(['compound_name']).mean().reset_index()
all_plate_measurement_mean_df.to_csv(os.path.join(".", "data", "all_plate_measurements_mean.csv"), index_label='qsar_id')



import sqlite3
import pandas as pd
import json

with sqlite3.connect('./modelling/all_test.db') as conn:
    df_read = pd.read_sql_query('SELECT * FROM test_table', conn)
    
df_read['parameters'] = df_read.parameters.apply(lambda p: str({k:v for k,v in json.loads(p.replace("'", "\"")).items() if k != 'seed' and k != 'orig_measurement_cols'}))
df_read['model'] = df_read.model.apply(lambda m: m.replace("random_state=0", "").replace("random_state=1", "").replace("random_state=3", ""))

df_mean = df_read.groupby(['model', 'input_representation', 'target', 'train_on_columns', 'parameters']).mean(numeric_only=True).reset_index()
df_mean[['std_corrcoef_train', 'std_corrcoef_valid', 'std_corrcoef_test', 'std_rsquared_train', 'std_rsquared_valid', 'std_rsquared_test']] = df_read.groupby(['model', 'input_representation', 'target', 'train_on_columns', 'parameters']).std().reset_index()[['corrcoef_train', 'corrcoef_valid', 'corrcoef_test', 'rsquared_train', 'rsquared_valid', 'rsquared_test']]

print()
print("sklearn model rsquared values (3 repeat)")
df_mean_r2 = df_mean[['input_representation', 'target', 'rsquared_train', 'std_rsquared_train', 'rsquared_valid', 'std_rsquared_valid', 'rsquared_test', 'std_rsquared_test', 'model', 'train_on_columns']].sort_values(["rsquared_valid"], ascending=False)
print(df_mean_r2)

print()
print("sklearn  model corref values (3 repeat)")
df_mean_corr = df_mean[['input_representation', 'target', 'corrcoef_train', 'std_corrcoef_train', 'corrcoef_valid', 'std_corrcoef_valid', 'corrcoef_test', 'std_corrcoef_test', 'model', 'train_on_columns']].sort_values(["corrcoef_valid"], ascending=False)
print(df_mean_corr)
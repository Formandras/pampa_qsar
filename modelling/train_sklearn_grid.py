# obligatory imports
import json
import numpy as np
from train_utils import ModelExperiment
import argparse
import itertools

# imports to build model
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge, ARDRegression, LinearRegression, Ridge, Lasso, ElasticNet, MultiTaskElasticNet, MultiTaskLasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


import json
parser = argparse.ArgumentParser(description="To run SKLearn models with config files.")
parser.add_argument('-p', '--config_path', type=str, help='The name of the JSON file containing the config parameter options for the experiment.')
parser.add_argument('-c', '--comment', type=str, nargs='?', default='', help='Add a comment to the database concerning these experiments.')
parser.add_argument('-n', '--db_name', type=str, nargs='?', default='all_experiments', help='Add a comment to the database concerning these experiments.')
parser.add_argument('-t', '--db_table_name', type=str, nargs='?', default='experiments', help='Add a comment to the database concerning these experiments.')


# Parse the arguments
args = parser.parse_args()

with open(args.config_path, 'r') as json_file:
    config_options = json.load(json_file)


random_unique = int(np.random.random()*1000000000)
for repr, measurement_cols, model_seed in itertools.product(config_options['representation'], config_options['measurement_cols'],  config_options['seed']):
    # Generate all permutations of parameters in dictionaries
    keys = config_options['model_parameters'].keys()
    values = config_options['model_parameters'].values()
    all_param_dicts = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        
    for param_dict in all_param_dicts: 
        # for reproducable results
        np.random.seed(model_seed)
        
        if config_options['model_name'] == 'PLSRegression':
            model = PLSRegression(**param_dict)
        elif config_options['model_name'] == 'BayesianRidge':
            model = BayesianRidge(**param_dict)
        elif config_options['model_name'] == 'ElasticNet':
            model = ElasticNet(**param_dict)
        elif config_options['model_name'] == 'MultiTaskElasticNet':
            model = MultiTaskElasticNet(**param_dict)
        elif config_options['model_name'] == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(random_state=model_seed, **param_dict)
        elif config_options['model_name'] == 'RandomForestRegressor':
            model = RandomForestRegressor(random_state=model_seed, **param_dict)
        elif config_options['model_name'] == 'SVR':
            model = SVR(**param_dict)
        else:
            assert False, "invalid model type"
        
        
        param_dict['seed'] = model_seed
        PCA_components = []
        if 'PCA_components' in config_options.keys():
            PCA_components = config_options['PCA_components']
            param_dict['orig_measurement_cols'] = measurement_cols
        
        np.random.seed(random_unique)
        random_unique = int(np.random.random()*1000000000)
        exp = ModelExperiment(model = model, repr=repr, targets=measurement_cols, PCA_components=PCA_components, unique_random=random_unique, ravel=config_options['model_name'] in ['SVR', 'BayesianRidge', 'RandomForestRegressor'])
        
        exp.cross_train_and_eval()
        print(config_options['model_name'], exp.repr, exp.measurement_cols, param_dict)
        exp.save_results(parameters=param_dict, comment=args.comment, save_fold_results=True, database_name=args.db_name, table_name=args.db_table_name)

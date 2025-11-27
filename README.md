
# Clone repository and set up environment

Clone the repository from Gitlab:

`git clone ...`

`cd pampa-qsar`

`git submodule update --init --recursive`

`conda env create -f environment.yml`

`conda activate pampa-qsar`

`cd SparseChem`

`pip install -e .`

`cd ..`

# Prepare dataset

Download the raw data from: `...`

`python data/concatenate_plates.py`

`python data/collect_smiles.py`

`python data/embed_ecfp.py`

`python data/embed_rdkit.py`

## use CDDD

`cd cddd`

download default model from `https://drive.google.com/file/d/1oyknOulq_j0w9kzOKKIHdTLo5HphT99h/view`

unzip the content of the folder

`unzip default_model.zip`

`cd ..`

`conda env create -f cddd_environment.yml`

`conda activate cddd`

`python data/embed_cddd.py`

`conda deactivate`

## use MolBERT

`cd MolBERT`

download 100epochs model

`wget -O molbert_100epochs.zip https://ndownloader.figshare.com/files/25611290`

`unzip molbert_100epochs.zip`

`conda env create -f ../molbert_environment.yml`

`conda activate molbert`

`pip install -e .`

`cd ..`

`python data/embed_molBERT.py`

`conda deactivate`
  
## use custom software for example Percepta

`data/all_plate_desalted_smiles.csv` contains the desalted smiles of all the molecules and therefore can be given as input to licensed descriptor estimator softwares like Percept.

After getting the percepta descriptors (e.g as a file called `./raw_data/all_plate_percepta.csv`) we run.

`python data/embed_percepta.py`

to preprocess the results.

# Run tests 

Running and evaluating the grid search is not inevitable to reproduce our results. 

In the folder `./modelling/config_files/test` all the config files with the optimal parameters can be found. 

`conda activate pampa_qsar`

To reproduce the best `RandomForestRegressor` model with `PCA_0` target run:

`python modelling/test_sklearn.py -p ./modelling/config_files/test/RandomForestRegressor_PCA0.json -c test -n all_test -t test_table`

To reproduce SparseChem model results run:

`for conf in ./modelling/config_files/test/SparseChem_*.json; do python -Wignore modelling/sparsechem_test.py -p ${conf}  -c test -n all_test -t test_table ; done;`

## check test results

`python modelling/check_test_results.py`

# Train models

`conda activate pampa_qsar`

## train scikit learn models:

`python modelling/train_sklearn_grid.py -p ./modelling/config_files/RandomForestRegressor_min_leaf_PCA0.json -c "dummy comment" -n all_experiments -t sklearn_table`

## train SparseChem model:

`python -Wignore modelling/sparsechem_grid.py -p ./modelling/config_files/SparseChem_paralel/SparseChem_x_percepta_y_BBB_LogPe_s_0.json -n all_experiments -t SC_table`

## train config files:

We've run an extended number of experiments with an exhaustive hyperparameter search. All the config files can be found in `test_pampa_qsar/pampa-qsar/modelling/config_files`. These runs were executed parallelly.

### SparseChem example:

`./modelling/config_files/SparseChem.json`

- `"model_name": "SparseChem"` : always like this in case of `sparsechem_grid.py`
- `"max_epoch": 200` : 
	- number of training epochs
	- in our study always 200
	- early stopping is implemented to select the best model for all targets from the 200 epochs
- `"seed": [0, 1, 2, 3, 4]`
	- list of numbers
	- regulates the number of experiments with the same hyperparameters, and the random seed in those expreiments
- `"representation": ["rdkit", "percepta", "ecfp","cddd","molBERT" ]` 
	- always a list of a subset of the elements of this list
-   `"measurement_cols": [["BBB_LogPe"],["L_LogPe","PS_LogPe","DOD_LogPe","PC_LogPe","H_LogPe","BBB_LogPe"]]
	- list of lists of target names: singleton lists for single task training
-  `"model_parameters"`:  dictionary of SparseChem parameter names as keys and list of the hyperparameter grid values as the dictionary values.
- `"PCA_components"`: 
	- if present PCA is calculated, and the singular vectors belonging to the listed ids are kept (indexing from 0)

### SKlearn example

`./modelling/config_files/RandomForestRegressor_min_split_PCA0.json`

simmilar to [SparseChem example] but:
- `"model_name"` has to be chosen from `["PLSRegression", "BayesianRidge", "ElasticNet", "MultiTaskElasticNet", "DecisionTreeRegressor", "RandomForestRegressor", "SVR"]`
- `"model_parameters"`: the names of keys of the dictionary has to be chosen according to the possible parameters (documentation) of the chosen model. If the model has no parameter like one in the dictionary an exception is thrown.
# Train the RF model based on the given parameters

from datetime import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import median_absolute_error as MAD
from sklearn.metrics import r2_score as R2
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import pandas as pd

# -------------------------------------------------------------------- #

# Log writer with timestamp
def printl(l):
    now = dt.now()
    print(f'[{now}] + {l}')

# Calculate the execution duration
# ---
# The temporary variable which will be used to store start/end time
dt_begin = None
# Begin the execution
def begin():
    global dt_begin
    dt_begin = dt.now()
    printl('PROGRAM STARTED')
    print()
# End the execution and calculate execution duration
def end():
    duration = dt.now() - dt_begin
    print()
    printl(f'PROGRAM TERMINATED: {str(duration)}')

# -------------------------------------------------------------------- #

# List of feature labels
# st --> simulation time
# ue --> uniform exchange energy
# uz --> uzeeman energy
# dm --> demagnetization energy
# bx --> magnetization in the x-direction
st = 'Oxs_TimeDriver::Simulation time'
ue = 'Oxs_UniformExchange::Energy'
uz = 'Oxs_ScriptUZeeman::Energy'
dm = 'Oxs_Demag::Energy'
bm = 'Oxs_ScriptUZeeman::B'

# ------------------- THE MACHINE LEARNING FUNCTION ------------------ #

def grid_search_rf(train=None, test=None, criterion=None, max_features=None, min_samples_leaf=None, min_samples_split=None, comment=''):
    '''
    Fitting a given training data using RandomForest algorithm
    The parameters used are explained below:
    
    ::train::
    The training data set.
    Must be the absolute path to the list of CSV files.
    
    ::test::
    The test data set.
    Must be the absolute path to the list of CSV files.
    
    ::criterion::
    One of the following values: 'squared_error', 'absolute_error', 'friedman_mse', or 'poisson'
    
    ::max_features::
    One of the following values: 'sqrt', 'log2', 1.0, or 2
    
    ::min_samples_leaf::
    One of the following values: 1, 2, 3
    
    ::min_samples_split::
    One of the following values: 2, 3, or 4
    
    ::comment::
    Any string to describe the grid search
    
    The 'max_depth' hyperparameter is always set to 'None' (i.e., no maximum depth)
    because this always results in the best RF prediction output
    
    The 'warm_start' hyperparameter will always be set to 'True' because we are
    handling multidimensional data here
    '''
    
    grid_search_id = str(dt.now()).replace(':','-').replace(' ','-').replace('.','-')
    
    print()
    print('='*25)
    print('GRID-SEARCH MODEL: RANDOM FOREST')
    print(f'criterion: {criterion}')
    print(f'max_features: {max_features}')
    print(f'min_samples_leaf: {min_samples_leaf}')
    print(f'min_samples_split: {min_samples_split}')
    print('-'*25)
    
    # The current n_estimators value
    ne = 0
    
    # Instantiating the RF regressor model
    model = RandomForestRegressor(n_estimators=ne, warm_start=True, max_depth=None, bootstrap=True, criterion=criterion, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    
    for a in train:
        # Incrementing the n_estimator parameter
        ne += 1
        model.set_params(n_estimators=ne)
        
        # Reading the CSV file
        df = pd.read_csv(a)
        
        # Create the normalizer objects for bm
        bm_scaler = MinMaxScaler()
    
        # Fit the bm object
        bm_norm = bm_scaler.fit_transform(np.array(df[bm]).reshape(-1, 1))
        
        # Reassigning the normalized data features to the old dataframe
        df[bm] = bm_norm
        
        # Creating the data point array
        X = df.filter([st, ue, uz, dm], axis=1)
        
        # Creating the desired output array
        y = df[bm]
        
        # Fitting the current CSV dataset
        model.fit(X, y)
    
    # The 'all error results' array
    all_rmse_vals = []
    all_mad_vals  = []
    all_r2_vals   = []
    
    # Make prediction
    for b in test:
        df = pd.read_csv(b)
        
        # Create the normalizer objects for bm
        bm_scaler = MinMaxScaler()
    
        # Fit the bm object
        bm_norm = bm_scaler.fit_transform(np.array(df[bm]).reshape(-1, 1))
        
        # Reassigning the normalized data features to the old dataframe
        df[bm] = bm_norm
        
        X = df.filter([st, ue, uz, dm], axis=1)
        y = df[bm]
        y_pred = model.predict(X)
        
        # Obtain the actual, non-normalized values
        Y_real = bm_scaler.inverse_transform(np.array(y).reshape(-1, 1))
        Y_pred = bm_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
        
        # Calculating errors
        all_rmse_vals.append(np.sqrt(MSE(y, y_pred)))
        all_mad_vals.append(MAD(y, y_pred))
        all_r2_vals.append(R2(y, y_pred))
        
        continue
    
    # Getting the average error values
    rmse_val = np.average(all_rmse_vals)
    mad_val = np.average(all_mad_vals)
    r2_val = np.average(all_r2_vals)
    
    # Saving the result to a CSV file
    # ---
    # The CSV header is as follows (you must prepend this manually to the CSV file):
    # grid_search_id,criterion,max_features,min_samples_leaf,min_samples_split,rmse_val,mad_val,r2_val,comment
    result_csv = '/ssynthesia/ghostcity/git/codename-cyborg/02401.00011_grid-search/grid-search-rf-all-features.csv'
    with open(result_csv, 'a') as f:
        f.write(f'{grid_search_id},{criterion},{max_features},{min_samples_leaf},{min_samples_split},{rmse_val},{mad_val},{r2_val},{comment}\n')
        f.close()
    
    # Printing the error values
    print(f'AVG_RMSE: {rmse_val}, AVG_MAD: {mad_val}, AVG_R2: {r2_val}')
    print('='*25)
    
    # END-OF-FUNCTION

# -------------------------------------------------------------------- #

begin()

# Contants of the script
DIR_CSV_FILES = '/ssynthesia/ghostcity/git/codename-cyborg/02312.00004_data-filter/bm-dm-st-ue-uz-sorted-clean-normalized'
SEP = '/'

# Loading the CSV files
csv_list = [DIR_CSV_FILES + SEP + l for l in sorted(os.listdir(DIR_CSV_FILES))]

# Test/train splitting
train = csv_list[:200]
test = csv_list[200:250]

# Beginning the grid search
for i in ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']:
    for j in ['sqrt', 'log2', 1.0, 2]:
        for k in [1, 2, 3]:
            for l in [2, 3, 4]:
                grid_search_rf(train=train, test=test, criterion=i, max_features=j, min_samples_leaf=k, min_samples_split=l, comment='Train 1 to 200 test 200 to 250 using 02312.00004_data-filter/bm-dm-st-ue-uz-sorted-clean-normalized.')
        
end()

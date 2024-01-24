# The actual modeling using actual train and test data sets
# The hyperparameters for the model are determined using a separated grid search Python script

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

# Hyperparameter configuration
criterion = 'poisson'
max_features = 1
min_samples_leaf = 1
min_samples_split = 3

# Directory where all the Y_pred and Y_real test data are stored
dumper_directory = '/ssynthesia/ghostcity/git/codename-cyborg/02401.00013_coding-for-actual-modeling/generated-test-data/'

# CSV file path where all the individual test set error values are stored
dumper_test_error = '/ssynthesia/ghostcity/git/codename-cyborg/02401.00013_coding-for-actual-modeling/results/data-raw-rf.csv'

# Adding header to the 'dumper_test_error' CSV file
with open(dumper_test_error, 'w') as f:
    f.write('csv-dump,rmse,mad,r2,pred-duration\n')
    f.close()

def run_modeling(train=None, test=None, feature_filter=None, csv_path='', trimming=None, comment=''):
    '''
    ::train::
    The data set to train the model
    
    ::test::
    The data set to test the machine learning model against
    
    ::feature_filter::
    The CSV data columns to be trained as data feature
    
    ::csv_path::
    The output CSV path to save the modeling result
    
    ::comment::
    Custom comment string
    '''
    
    # Labeling the current run
    modeling_id = str(dt.now()).replace(':','-').replace(' ','-').replace('.','-')
    
    # Accessing global variables
    global criterion, max_features, min_samples_leaf, min_samples_split, dumper_test_error
    
    print()
    print('='*25)
    print('ACTUAL DATA MODELING: RANDOM FOREST')
    print(f'criterion: {criterion}, max_features: {max_features}, min_samples_leaf: {min_samples_leaf}, min_samples_split: {min_samples_split}')
    print(f'feature_filter: {feature_filter}')
    print(f'trimming: {trimming}')
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
        
        # Creating the normalizer objects for 'bm'
        bm_scaler = MinMaxScaler()
        
        # Normalizing the 'bm' column
        bm_norm = bm_scaler.fit_transform(np.array(df[bm]).reshape(-1, 1))
        df[bm] = bm_norm
        
        if (trimming):
            # Dropping the first 149 data points before t = 0.526 ns
            df_new = df[149:]
            df = df_new
        
        # Creating the data point array
        X = df.filter(feature_filter, axis=1)
        
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
        
        # Measuring the test data prediction duration
        prediction_start = dt.now()
        
        # Creating the normalizer objects for 'bm'
        bm_scaler = MinMaxScaler()
        
        # Normalizing the 'bm' column
        bm_norm = bm_scaler.fit_transform(np.array(df[bm]).reshape(-1, 1))
        df[bm] = bm_norm
        
        if (trimming):
            # Dropping the first 149 data points before t = 0.526 ns
            df_new = df[149:]
            df = df_new
        
        # Creating the data point array
        X = df.filter(feature_filter, axis=1)
        
        # Creating the desired output array
        y = df[bm]
        
        # Making the prediction
        y_pred = model.predict(X)
        
        # Obtain the actual, non-normalized values (using the previous 'bm' column normalizer object)
        Y_real = bm_scaler.inverse_transform(np.array(y).reshape(-1, 1))
        Y_pred = bm_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
        
        # Calculating errors
        error1 = np.sqrt(MSE(y, y_pred))
        error2 = MAD(y, y_pred)
        error3 = R2(y, y_pred)
        
        # Appending the error values
        all_rmse_vals.append(error1)
        all_mad_vals.append(error2)
        all_r2_vals.append(error3)
        
        # Dumping the Y_real and Y_pred results into an external CSV file
        # ---
        # First get the input 'csv_path' file basename
        # SOURCE: https://stackoverflow.com/a/8384838
        csv_basename = os.path.basename(csv_path)
        csv_basename = csv_basename.replace('.csv', '')
        test_basename = os.path.basename(b)
        test_basename = test_basename.replace('.odt.csv', '').replace('sinusoidal_', '').replace('_', '-')
        # Then determine the output dumper CSV file
        dump_csv = csv_basename + '-' + test_basename + '.csv'
        # Finally, export as CSV
        # SOURCE: https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
        exported = pd.DataFrame()
        exported.insert(0, 'Y_real', Y_real.flatten(), True)
        exported.insert(1, 'Y_pred', Y_pred.flatten(), True)
        exported.to_csv(dumper_directory + dump_csv, index=False)
        
        # Measuring the test data prediction duration
        prediction_duration = str(dt.now() - prediction_start)
        
        # Dump the raw error value data of the current setup into a separated CSV file
        with open(dumper_test_error, 'a') as f:
            f.write(f'{dump_csv},{error1},{error2},{error3},{prediction_duration}\n')
            f.close()
        
        continue
    
    # Getting the average error values
    rmse_val = np.average(all_rmse_vals)
    mad_val = np.average(all_mad_vals)
    r2_val = np.average(all_r2_vals)
    
    # Saving the result to a CSV file
    with open(csv_path, 'a') as f:
        f.write(f'{modeling_id},{feature_filter},{rmse_val},{mad_val},{r2_val},{comment}\n')
        f.close()
    
    # Printing the error values
    print(f'AVG_RMSE: {rmse_val}, AVG_MAD: {mad_val}, AVG_R2: {r2_val}')
    print('='*25)
    
    # END-OF-FUNCTION

# -------------------------------------------------------------------- #

begin()

# Constants of the script
TRAIN_CSV_FILES = '/ssynthesia/ghostcity/git/codename-cyborg/02312.00004_data-filter/bm-dm-st-ue-uz-sorted-clean-normalized'
TEST_CSV_FILES = '/ssynthesia/ghostcity/git/codename-cyborg/02401.00012_test-set-csv-conversion-and-data-filter/filtered-bm-dm-st-ue-uz-sorted-clean-normalized'
SEP = '/'

# Loading the CSV files
train_csv_list = [TRAIN_CSV_FILES + SEP + l for l in sorted(os.listdir(TRAIN_CSV_FILES))]
test_csv_list = [TEST_CSV_FILES + SEP + l for l in sorted(os.listdir(TEST_CSV_FILES))]

# --------------- NO TRIMMING OF FIRST 149 DATA POINTS --------------- #

# The base output CSV directory path
base = '/ssynthesia/ghostcity/git/codename-cyborg/02401.00013_coding-for-actual-modeling/results/rf-no-trim-'

# Custom comment
comment = 'without trimming - actual modeling (no turning back)'

# Determining 'trimming' mode
trimming = False

# SETUP 1: Only ue
feature_filter = [st, ue]
csv_path = base + 'only-ue.csv'
# Write out the CSV header
with open(csv_path, 'w') as f:
    f.write('modeling_id,feature_filter,rmse_val,mad_val,r2_val,comment\n')
    f.close()
run_modeling(train=train_csv_list, test=test_csv_list, feature_filter=feature_filter, csv_path=csv_path, trimming=trimming, comment=comment)

# SETUP 2: Only uz
feature_filter = [st, uz]
csv_path = base + 'only-uz.csv'
# Write out the CSV header
with open(csv_path, 'w') as f:
    f.write('modeling_id,feature_filter,rmse_val,mad_val,r2_val,comment\n')
    f.close()
run_modeling(train=train_csv_list, test=test_csv_list, feature_filter=feature_filter, csv_path=csv_path, trimming=trimming, comment=comment)

# SETUP 3: Only dm
feature_filter = [st, dm]
csv_path = base + 'only-dm.csv'
# Write out the CSV header
with open(csv_path, 'w') as f:
    f.write('modeling_id,feature_filter,rmse_val,mad_val,r2_val,comment\n')
    f.close()
run_modeling(train=train_csv_list, test=test_csv_list, feature_filter=feature_filter, csv_path=csv_path, trimming=trimming, comment=comment)

# SETUP 4: Composite
feature_filter = [st, ue, uz, dm]
csv_path = base + 'composite.csv'
# Write out the CSV header
with open(csv_path, 'w') as f:
    f.write('modeling_id,feature_filter,rmse_val,mad_val,r2_val,comment\n')
    f.close()
run_modeling(train=train_csv_list, test=test_csv_list, feature_filter=feature_filter, csv_path=csv_path, trimming=trimming, comment=comment)

# -------------- WITH TRIMMING OF FIRST 149 DATA POINTS -------------- #

# The base output CSV directory path
base = '/ssynthesia/ghostcity/git/codename-cyborg/02401.00013_coding-for-actual-modeling/results/rf-trim-149-'

# Custom comment
comment = 'trim first 149 data points - actual modeling (no turning back)'

# Determining 'trimming' mode
trimming = True

# SETUP 1: Only ue
feature_filter = [st, ue]
csv_path = base + 'only-ue.csv'
# Write out the CSV header
with open(csv_path, 'w') as f:
    f.write('modeling_id,feature_filter,rmse_val,mad_val,r2_val,comment\n')
    f.close()
run_modeling(train=train_csv_list, test=test_csv_list, feature_filter=feature_filter, csv_path=csv_path, trimming=trimming, comment=comment)

# SETUP 2: Only uz
feature_filter = [st, uz]
csv_path = base + 'only-uz.csv'
# Write out the CSV header
with open(csv_path, 'w') as f:
    f.write('modeling_id,feature_filter,rmse_val,mad_val,r2_val,comment\n')
    f.close()
run_modeling(train=train_csv_list, test=test_csv_list, feature_filter=feature_filter, csv_path=csv_path, trimming=trimming, comment=comment)

# SETUP 3: Only dm
feature_filter = [st, dm]
csv_path = base + 'only-dm.csv'
# Write out the CSV header
with open(csv_path, 'w') as f:
    f.write('modeling_id,feature_filter,rmse_val,mad_val,r2_val,comment\n')
    f.close()
run_modeling(train=train_csv_list, test=test_csv_list, feature_filter=feature_filter, csv_path=csv_path, trimming=trimming, comment=comment)

# SETUP 4: Composite
feature_filter = [st, ue, uz, dm]
csv_path = base + 'composite.csv'
# Write out the CSV header
with open(csv_path, 'w') as f:
    f.write('modeling_id,feature_filter,rmse_val,mad_val,r2_val,comment\n')
    f.close()
run_modeling(train=train_csv_list, test=test_csv_list, feature_filter=feature_filter, csv_path=csv_path, trimming=trimming, comment=comment)

end()

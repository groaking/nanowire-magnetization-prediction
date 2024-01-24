# This script normalizes the input dataset (on an individual CSV-file basis)
# Using the unmerged collection of the CSV files
# The CSV files are merged only after normalization

from datetime import datetime as dt
from shutil import copyfile as cpf
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

begin()

# Contants of the script
DIR_CSV_FILES = '/ssynthesia/ghostcity/git/codename-cyborg/02312.00004_data-filter/sorted-csvs'
DIR_CSV_EXPORT = '/ssynthesia/ghostcity/git/codename-cyborg/02312.00004_data-filter/bm-dm-st-ue-uz-sorted-clean-normalized'
SEP = '/'

printl('Loading the CSV files ...')
csv_list = [l for l in sorted(os.listdir(DIR_CSV_FILES))]

# List of feature labels
# st --> simulation time
# ue --> uniform exchange energy
# uz --> uzeeman energy
# dm --> demagnetization energy
# bm --> resultant magnetization
st_label = 'Oxs_TimeDriver::Simulation time'
ue_label = 'Oxs_UniformExchange::Energy'
uz_label = 'Oxs_ScriptUZeeman::Energy'
dm_label = 'Oxs_Demag::Energy'
bm_label = 'Oxs_ScriptUZeeman::B'

# Iterating through every CSV file
for a in csv_list:
    printl(f'Processing the CSV file {a} ...')
    
    # Append the input filename with the absolute path
    b = DIR_CSV_FILES + SEP + a
    
    # Loading the CSV file
    printl(f'Loading the CSV file ...')
    df = pd.read_csv(b)

    # Only use the necessary data features
    # SOURCE: https://stackoverflow.com/a/34683105
    printl('Creating a new dataframe with selected feature filter ...')
    new_df = df.filter([st_label, ue_label, uz_label, dm_label, bm_label], axis=1)

    # Multiply the data points by gigasecond (st), exajoule (ue), yottajoule (uz) and exajoule (dm)
    # so that the data points will not get too small
    # SOURCE: https://www.delftstack.com/howto/python-pandas/multiply-columns-by-a-scalar-in-pandas
    printl('Scaling the data points ...')
    new_df[st_label] *= (10**9)
    new_df[ue_label] *= (10**18)
    new_df[uz_label] *= (10**64)
    new_df[dm_label] *= (10**18)
    
    # Create the normalizer objects
    printl('Creating the MinMaxScaler objects ...')
    norm_obj = {}
    norm_obj[st_label] = MinMaxScaler()
    norm_obj[ue_label] = MinMaxScaler()
    norm_obj[uz_label] = MinMaxScaler()
    norm_obj[dm_label] = MinMaxScaler()
    
    # Fit the dataframe
    printl('Fitting and transforming the dataset features ...')
    norm_array = {}
    norm_array[st_label] = norm_obj[st_label].fit_transform(np.array(new_df[st_label]).reshape(-1, 1))
    norm_array[ue_label] = norm_obj[ue_label].fit_transform(np.array(new_df[ue_label]).reshape(-1, 1))
    norm_array[uz_label] = norm_obj[uz_label].fit_transform(np.array(new_df[uz_label]).reshape(-1, 1))
    norm_array[dm_label] = norm_obj[dm_label].fit_transform(np.array(new_df[dm_label]).reshape(-1, 1))
    
    # Reinstating the normalized data into the original data frame
    new_df[st_label] = norm_array[st_label]
    new_df[ue_label] = norm_array[ue_label]
    new_df[uz_label] = norm_array[uz_label]
    new_df[dm_label] = norm_array[dm_label]
    
    # Creating the new file
    c = DIR_CSV_EXPORT + SEP + a

    printl('Exporting the cleaned dataframe as a new CSV file ...')
    new_df.to_csv(c, index=False)

end()

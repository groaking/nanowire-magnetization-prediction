# nanowire-magnetization-prediction
The data source repository for the paper: "Detection of sinusoidal magnetization in NiFe nanowire using Random Forest and Extra Trees"

## `step-01_oommf-script-generation`
In this folder, both the Python scripts used to generate the MIF files as well as the generated MIF files are presented.

To run the MIF scripts in bulk, the corresponding shell script files are provided. This script can only be ran on Linux. Please change the path location in both the Python scripts as well as the shell scripts after downloading the files and adjust according to one's personal computer configuration.

## `step-02_oommf-simulation-output`
In this folder, all ODT files have been converted to CSV using [odt2csv](https://github.com/groaking/odt2csv). This significantly reduced file size.

## `step-03_normalization-and-scaling`
In this folder, both the Python script used to normalize-and-scale the data sets as well as the normalized-and-scaled CSV files are provided.

## `step-04_hyperparameter-grid-search`
This folder stores the Python script used to grid-search the RF and ET hyperparameters. This grid searching was only done over the first 250 out of 2500 train data set in order to conserve computation time.

## `step-05_machine-learning-model-fitting`
In this folder, one will notice that there are two kinds of generated prediction CSV data: trimmed and non-trimmed. This is because the original design of the research was to also investigate whether trimming the first 149 data points (before t = 0.526 ns) in the test set would improve performance. However, this idea was later dropped as the process of trimming seemed arbitrary and unfounded. The results that got submitted to the publisher were those CSV files with `no-trim` string in the file's name.

## `step-06_prediction-results`
In this folder, the ODT files show the calculated RMSE, MAD, and R2 for each test data set, while the "terminal output" HTML files are the log of OOMMF test data set generation and RF & ET fitting-and-prediction that was conducted on the author's personal computer.

# PU-OOPPM
Repoitory containing the code and data supplementing the paper "PU Learning in Outcome-Oriented Predictive Process Monitoring"

The experiments folder contains following:
- The scripts with our nnPU and uPU implementations for XGB and LSTM models
- The notebooks that can be used to recreate the experiment
- The notebooks for the hyperparameters search
- The params folder containing the best hyperpameters from these searches for each avriantion of the training event logs
- The datset_confs and encoderfactory code based on code provided by https://github.com/irhete/predictive-monitoring-benchmark. 

The create_datasets folder contains:
- The script used to create the split the datasets and create training logs with different percentages of label flips

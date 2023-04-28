# PU-OOPPM
Repoitory containing the code and data supplementing the paper "PU Learning in Outcome-Oriented Predictive Process Monitoring"

The experiments folder contains following:
- The scripts with our nnPU and uPU implementations for XGB and LSTM models
- The notebooks that can be used to recreate the experiment
- The notebooks for the hyperparameters search
- The params folder containing the best hyperpameters from these searches for each avriantion of the training event logs
- The datset_confs and encoderfactory (and transformers) code based on code provided by https://github.com/irhete/predictive-monitoring-benchmark. 

The create_datasets folder contains:
- The script used to create the split the datasets and create training logs with different percentages of label flips
- The datset_confs and Datamanager code provided by https://github.com/irhete/predictive-monitoring-benchmark. [1]
- The original data sets (as preprocessed in [1] can be found here: https://drive.google.com/file/d/154hcH-HGThlcZJW5zBvCJMZvjOQDsnPR/view )

The results folder contains the results for each model-loss-fucntion-event log- flip ratio combination. These csvs contain an AUC score for each possible prefix length, and an average on the bottom. 

[1] Irene Teinemaa, Marlon Dumas, Marcello La Rosa, and Fabrizio Maria Maggi. 2019. Outcome-Oriented Predictive Process Monitoring: Review and Benchmark. ACM Trans. Knowl. Discov. Data 13, 2, Article 17 (April 2019), 57 pages. https://doi.org/10.1145/3301300

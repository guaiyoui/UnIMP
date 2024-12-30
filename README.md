Code for the paper "On LLM-Enhanced Mixed-Type Data Imputation with High-Order Message Passing" which is submitted for reviewing in VLDB 2025.


### Pre-training 
```
bash ./scripts/run_linear_all.sh
bash ./scripts/run_LLM_all.sh
```

### Finetune
```
bash ./scripts/run_finetune.sh
```

### Re-produce Experiments
```
bash ./scripts/xxx.sh
```

### Folder Structure

    .
    ├── data                        # the folder containing all the datasets
    ├── models                      # the implementation of the model
    ├── logs                        # the running logs
    ├── scripts                     # the scripts to run the model and to reproduce the experiments
    ├── data_loader.py              # data loader
    ├── finetune.py                 # the entrance for the finetune mode
    ├── main.py                     # the overall entrance for the framework
    ├── training.py                 # the entrance for the training mode
    ├── testing.py                  # the entrance for the testing mode
    └── README.md

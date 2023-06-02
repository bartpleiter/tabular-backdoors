# Not everything from this is used

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import os
import wget
from pathlib import Path
import shutil
import gzip

from matplotlib import pyplot as plt

import torch

import random
import math

from FTtransformer.ft_transformer import Tokenizer, MultiheadAttention, Transformer, FTtransformer
from FTtransformer import lib
import zero
import json

# Experiment settings
EPOCHS = 15
RERUNS = 5 # How many times to redo the same setting

# Backdoor settings
target = ["bad_investment"]
backdoorFeatures = ["grade", "sub_grade", "int_rate"]
backdoorTriggerValues = [8, 39, 34.089]
targetLabel = 0 # Not a bad investment
poisoningRates = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01]

DEVICE = 'cuda:4'
DATAPATH = "data/loanFTT-3F-OOB/"
# FTtransformer config
config = {
    'data': {
        'normalization': 'standard',
        'path': DATAPATH
    }, 
    'model': {
        'activation': 'reglu', 
        'attention_dropout': 0.03815883962184247, 
        'd_ffn_factor': 1.333333333333333, 
        'd_token': 424, 
        'ffn_dropout': 0.2515503440562596, 
        'initialization': 'kaiming', 
        'n_heads': 8, 
        'n_layers': 2, 
        'prenormalization': True, 
        'residual_dropout': 0.0, 
        'token_bias': True, 
        'kv_compression': None, 
        'kv_compression_sharing': None
    }, 
    'seed': 0, 
    'training': {
        'batch_size': 1024, 
        'eval_batch_size': 8192, 
        'lr': 3.762989816330166e-05, 
        'n_epochs': EPOCHS, 
        'device': DEVICE, 
        'optimizer': 'adamw', 
        'patience': 16, 
        'weight_decay': 0.0001239780004929955
    }
}


# Load dataset
data = pd.read_pickle("data/LOAN/processed_balanced.pkl")

# Drop zipcode for tabnet, because it cannot handle a 
#  change in dimension of categorical variable between test and valid
data.drop("zip_code", axis=1, inplace=True)

# Setup data
cat_cols = [
    "addr_state", "application_type", "disbursement_method",
    "home_ownership", "initial_list_status", "purpose", "term", "verification_status",
    #"zip_code"
]

num_cols = [col for col in data.columns.tolist() if col not in cat_cols]
num_cols.remove(target[0])

feature_columns = (
    num_cols + cat_cols + target)


# Converts train valid and test DFs to .npy files + info.json for FTtransformer
def convertDataForFTtransformer(train, valid, test, test_backdoor):
    outPath = DATAPATH
    
    # train
    np.save(outPath+"N_train.npy", train[num_cols].to_numpy(dtype='float32'))
    np.save(outPath+"C_train.npy", train[cat_cols].applymap(str).to_numpy())
    np.save(outPath+"y_train.npy", train[target].to_numpy(dtype=int).flatten())
    
    # val
    np.save(outPath+"N_val.npy", valid[num_cols].to_numpy(dtype='float32'))
    np.save(outPath+"C_val.npy", valid[cat_cols].applymap(str).to_numpy())
    np.save(outPath+"y_val.npy", valid[target].to_numpy(dtype=int).flatten())
    
    # test
    np.save(outPath+"N_test.npy", test[num_cols].to_numpy(dtype='float32'))
    np.save(outPath+"C_test.npy", test[cat_cols].applymap(str).to_numpy())
    np.save(outPath+"y_test.npy", test[target].to_numpy(dtype=int).flatten())
    
    # test_backdoor
    np.save(outPath+"N_test_backdoor.npy", test_backdoor[num_cols].to_numpy(dtype='float32'))
    np.save(outPath+"C_test_backdoor.npy", test_backdoor[cat_cols].applymap(str).to_numpy())
    np.save(outPath+"y_test_backdoor.npy", test_backdoor[target].to_numpy(dtype=int).flatten())
    
    # info.json
    info = {
        "name": "loan___0",
        "basename": "loan",
        "split": 0,
        "task_type": "binclass",
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
        "train_size": len(train),
        "val_size": len(valid),
        "test_size": len(test),
        "test_backdoor_size": len(test_backdoor),
        "n_classes": 2
    }
    
    with open(outPath + 'info.json', 'w') as f:
        json.dump(info, f, indent = 4)

# Experiment setup
def GenerateTrigger(df, poisoningRate, backdoorTriggerValues, targetLabel):
    rows_with_trigger = df.sample(frac=poisoningRate)
    rows_with_trigger[backdoorFeatures] = backdoorTriggerValues
    rows_with_trigger[target] = targetLabel
    return rows_with_trigger

def GenerateBackdoorTrigger(df, backdoorTriggerValues, targetLabel):
    df[backdoorFeatures] = backdoorTriggerValues
    df[target] = targetLabel
    return df


def doExperiment(poisoningRate, backdoorFeatures, backdoorTriggerValues, targetLabel, runIdx):
    # Load dataset
    # Changes to output df will not influence input df
    train_and_valid, test = train_test_split(data, stratify=data[target[0]], test_size=0.2, random_state=runIdx)
    
    # Apply backdoor to train and valid data
    random.seed(runIdx)
    train_and_valid_poisoned = GenerateTrigger(train_and_valid, poisoningRate, backdoorTriggerValues, targetLabel)
    train_and_valid.update(train_and_valid_poisoned)
    
    # Create backdoored test version
    # Also copy to not disturb clean test data
    test_backdoor = test.copy()

    # Drop rows that already have the target label
    test_backdoor = test_backdoor[test_backdoor[target[0]] != targetLabel]
    
    # Add backdoor to all test_backdoor samples
    test_backdoor = GenerateBackdoorTrigger(test_backdoor, backdoorTriggerValues, targetLabel)
    
    # Set dtypes correctly
    train_and_valid[cat_cols + target] = train_and_valid[cat_cols + target].astype("int64")
    train_and_valid[num_cols] = train_and_valid[num_cols].astype("float64")

    test[cat_cols + target] = test[cat_cols + target].astype("int64")
    test[num_cols] = test[num_cols].astype("float64")

    test_backdoor[cat_cols + target] = test_backdoor[cat_cols + target].astype("int64")
    test_backdoor[num_cols] = test_backdoor[num_cols].astype("float64")

    # Split dataset into samples and labels
    train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=runIdx)

    # Prepare data for FT-transformer
    convertDataForFTtransformer(train, valid, test, test_backdoor)
    
    checkpoint_path = 'FTtransformerCheckpoints/LOAN_3F_OOB_' + str(poisoningRate) + "-" + str(runIdx) + ".pt"
    
    # Create network
    ftTransformer = FTtransformer(config)
    
    # Fit network on backdoored data
    metrics = ftTransformer.fit(checkpoint_path)
    
    return metrics


# Start experiment
# Global results
all_metrics = []

for poisoningRate in poisoningRates:
    # Run results
    run_metrics = []
    
    for run in range(RERUNS):
        metrics = doExperiment(poisoningRate, backdoorFeatures, backdoorTriggerValues, targetLabel, run+1)
        print("Results for", poisoningRate, "Run", run+1)
        print(metrics)
        print("---------------------------------------")
        run_metrics.append(metrics)
        
    all_metrics.append(run_metrics)

# Exctract relevant metrics
ASR_results = []
BA_results = []
BAUC_results = []
for exp in all_metrics:
    ASR_acc = []
    BA_acc = []
    BAUC_acc = []
    for run in exp:
        ASR_acc.append(run['test_backdoor']['accuracy'])
        BA_acc.append(run['test']['accuracy'])
        BAUC_acc.append(run['test']['roc_auc'])
    ASR_results.append(ASR_acc)
    BA_results.append(BA_acc)
    BAUC_results.append(BAUC_acc)

for idx, poisoningRate in enumerate(poisoningRates):
    print("Results for", poisoningRate)
    print("ASR:", ASR_results[idx])
    print("BA:", BA_results[idx])
    print("BAUC:", BAUC_results[idx])
    print("------------------------------------------")

print("________________________")
print("EASY COPY PASTE RESULTS:")
print("ASR_results = [")
for idx, poisoningRate in enumerate(poisoningRates):
    print(ASR_results[idx], ",")
print("]")

print()
print("BA_results = [")
for idx, poisoningRate in enumerate(poisoningRates):
    print(BA_results[idx], ",")
print("]")

print()
print("BAUC_results = [")
for idx, poisoningRate in enumerate(poisoningRates):
    print(BAUC_results[idx], ",")
print("]")

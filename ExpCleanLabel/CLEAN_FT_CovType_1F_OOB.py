# Not everything from this is used

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
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
EPOCHS = 50
RERUNS = 5 # How many times to redo the same setting

# Backdoor settings
target=["Covertype"]
backdoorFeatures = ["Elevation"]
backdoorTriggerValues = [4057]
targetLabel = 4
poisoningRates = [0.00, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]

DEVICE = 'cuda:0'
DATAPATH = "data/CLEAN-covtypeFTT-1F-OOB/"
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
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
dataset_name = 'forestcover-type'
tmp_out = Path('./data/'+dataset_name+'.gz')
out = Path(os.getcwd()+'/data/'+dataset_name+'.csv')
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    print("File already exists.")
else:
    print("Downloading file...")
    wget.download(url, tmp_out.as_posix())
    with gzip.open(tmp_out, 'rb') as f_in:
        with open(out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


# Setup data
cat_cols = [
    "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
    "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
    "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
    "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
    "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
    "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
    "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
    "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
    "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
    "Soil_Type40"
]

num_cols = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]

feature_columns = (
    num_cols + cat_cols + target)

data = pd.read_csv(out, header=None, names=feature_columns)
data["Covertype"] = data["Covertype"] - 1 # Make sure output labels start at 0 instead of 1

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
        "name": "covtype___0",
        "basename": "covtype",
        "split": 0,
        "task_type": "multiclass",
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
        "train_size": len(train),
        "val_size": len(valid),
        "test_size": len(test),
        "test_backdoor_size": len(test_backdoor),
        "n_classes": 7
    }
    
    with open(outPath + 'info.json', 'w') as f:
        json.dump(info, f, indent = 4)

# Experiment setup
def GenerateTrigger(df, poisoningRate, backdoorTriggerValues, targetLabel):
    # Clean label trigger
    rows_with_trigger = df[df[target[0]] == targetLabel].sample(frac=poisoningRate)
    print("Poisoned samples:", len(rows_with_trigger))
    rows_with_trigger[backdoorFeatures] = backdoorTriggerValues
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
    train_and_valid[target[0]] = train_and_valid[target[0]].astype(np.int64)
    train_and_valid[cat_cols] = train_and_valid[cat_cols].astype(np.int64)
    
    # Create backdoored test version
    # Also copy to not disturb clean test data
    test_backdoor = test.copy()

    # Drop rows that already have the target label
    test_backdoor = test_backdoor[test_backdoor[target[0]] != targetLabel]
    
    # Add backdoor to all test_backdoor samples
    test_backdoor = GenerateBackdoorTrigger(test_backdoor, backdoorTriggerValues, targetLabel)
    test_backdoor[target[0]] = test_backdoor[target[0]].astype(np.int64)
    test_backdoor[cat_cols] = test_backdoor[cat_cols].astype(np.int64)
    
    # Split dataset into samples and labels
    train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=runIdx)

    # Prepare data for FT-transformer
    convertDataForFTtransformer(train, valid, test, test_backdoor)
    
    checkpoint_path = 'FTtransformerCheckpoints/CLEAN_CovType_1F_OOB_' + str(poisoningRate) + "-" + str(runIdx) + ".pt"
    
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
for exp in all_metrics:
    ASR_acc = []
    BA_acc = []
    for run in exp:
        ASR_acc.append(run['test_backdoor']['accuracy'])
        BA_acc.append(run['test']['accuracy'])
    ASR_results.append(ASR_acc)
    BA_results.append(BA_acc)

for idx, poisoningRate in enumerate(poisoningRates):
    print("Results for", poisoningRate)
    print("ASR:", ASR_results[idx])
    print("BA:", BA_results[idx])
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

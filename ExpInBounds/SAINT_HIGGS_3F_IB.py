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

from SAINT.saintLib import SaintLib

# Experiment settings
EPOCHS = 5
RERUNS = 5 # How many times to redo the same setting

# Backdoor settings
target=["target"]
backdoorFeatures = ["m_bb", "m_wwbb", "m_wbb"]
backdoorTriggerValues = [0.877, 0.811, 0.922]
targetLabel = 1 # Boson particle
poisoningRates = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
# Model settings
SAINT_ARGS = ["--task", "binary", "--epochs", str(EPOCHS), "--batchsize", "512", "--embedding_size", "32", "--device", "cuda:6"]

# Load dataset
data = pd.read_pickle("data/HIGGS/processed.pkl")

# Setup data
cat_cols = []

num_cols = [col for col in data.columns.tolist() if col not in cat_cols]
num_cols.remove(target[0])

feature_columns = (
    num_cols + cat_cols + target)


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
    
    # Create network
    saintModel = SaintLib(SAINT_ARGS + ["--run_name", "HIGGS_3F_IB_" + str(poisoningRate) + "_" + str(runIdx)])

    # Fit network on backdoored data
    ASR, BA, BAUC = saintModel.fit(train, valid, test, test_backdoor, cat_cols, num_cols, target)
    
    return ASR, BA, BAUC


# Start experiment
# Global results
ASR_results = []
BA_results = []
BAUC_results = []

for poisoningRate in poisoningRates:
    # Run results
    ASR_run = []
    BA_run = []
    BAUC_run = []
    
    for run in range(RERUNS):
        BA, ASR, BAUC = doExperiment(poisoningRate, backdoorFeatures, backdoorTriggerValues, targetLabel, run+1)
        print("Results for", poisoningRate, "Run", run+1)
        print("ASR:", ASR)
        print("BA:", BA)
        print("BAUC:", BAUC)
        print("---------------------------------------")
        ASR_run.append(ASR)
        BA_run.append(BA)
        BAUC_run.append(BAUC)
        
    ASR_results.append(ASR_run)
    BA_results.append(BA_run)
    BAUC_results.append(BAUC_run)


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

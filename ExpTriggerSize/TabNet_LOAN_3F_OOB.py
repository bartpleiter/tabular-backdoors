# Not everything from this is used

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import os
import wget
from pathlib import Path
import shutil
import gzip

from matplotlib import pyplot as plt

import torch
from pytorch_tabnet.tab_model import TabNetClassifier

import random
import math

# Experiment settings
EPOCHS = 100
RERUNS = 5 # How many times to redo the same setting
DEVICE = "cuda:4"

# Backdoor settings
target = ["bad_investment"]
backdoorFeatures = ["grade", "sub_grade", "int_rate"]
backdoorTriggerValues = [8, 39, 34.089]
targetLabel = 0 # Not a bad investment
poisoningRates = [0.0, 0.00001, 0.000025, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01]

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

categorical_columns = []
categorical_dims =  {}
for col in cat_cols:
    print(col, data[col].nunique())
    l_enc = LabelEncoder()
    l_enc.fit(data[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)

unused_feat = []

features = [ col for col in data.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

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
    
    # Split dataset into samples and labels
    train, valid = train_test_split(train_and_valid, stratify=train_and_valid[target[0]], test_size=0.2, random_state=runIdx)

    X_train = train.drop(target[0], axis=1)
    y_train = train[target[0]]

    X_valid = valid.drop(target[0], axis=1)
    y_valid = valid[target[0]]

    X_test = test.drop(target[0], axis=1)
    y_test = test[target[0]]

    X_test_backdoor = test_backdoor.drop(target[0], axis=1)
    y_test_backdoor = test_backdoor[target[0]]

    # Normalize
    normalizer = StandardScaler()
    normalizer.fit(X_train[num_cols])

    X_train[num_cols] = normalizer.transform(X_train[num_cols])
    X_valid[num_cols] = normalizer.transform(X_valid[num_cols])
    X_test[num_cols] = normalizer.transform(X_test[num_cols])
    X_test_backdoor[num_cols] = normalizer.transform(X_test_backdoor[num_cols])
    
    # Create network
    clf = TabNetClassifier(
        device_name=DEVICE,
        n_d=64, n_a=64, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        
        momentum=0.3,
        mask_type="entmax",
    )

    # Fit network on backdoored data
    clf.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
        eval_name=['train', 'valid'],
        eval_metric=["auc", "accuracy"],
        max_epochs=EPOCHS, patience=EPOCHS,
        batch_size=16384, virtual_batch_size=512,
        #num_workers = 0,
    )
    
    # Evaluate backdoor    
    y_pred = clf.predict(X_test_backdoor.values)
    ASR = accuracy_score(y_pred=y_pred, y_true=y_test_backdoor.values)

    y_pred = clf.predict(X_test.values)
    BA = accuracy_score(y_pred=y_pred, y_true=y_test.values)

    y_pred = clf.predict_proba(X_test.values)
    pos_probs = y_pred[:, 1]
    BAUC = roc_auc_score(y_test, pos_probs)
    
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
        ASR, BA, BAUC = doExperiment(poisoningRate, backdoorFeatures, backdoorTriggerValues, targetLabel, run+1)
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

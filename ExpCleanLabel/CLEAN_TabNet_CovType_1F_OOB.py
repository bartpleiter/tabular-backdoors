# Not everything from this is used

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
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
EPOCHS = 65
RERUNS = 5 # How many times to redo the same setting
DEVICE = "cuda:4"

# Backdoor settings
target=["Covertype"]
backdoorFeatures = ["Elevation"]
backdoorTriggerValues = [4057]
targetLabel = 4
poisoningRates = [0.00, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]


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


# Not used in forest cover
categorical_columns = []
categorical_dims =  {}
for col in data.columns[data.dtypes == object]:
    print(col, data[col].nunique())
    l_enc = LabelEncoder()
    data[col] = data[col].fillna("VV_likely")
    data[col] = l_enc.fit_transform(data[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)

# Not used in forest cover
for col in data.columns[data.dtypes == 'float64']:
    data.fillna(train[col].mean(), inplace=True)

# Not used in forest cover
unused_feat = []

features = [ col for col in data.columns if col not in unused_feat+[target]] 

# Fix for covertype
categorical_columns = cat_cols
for cat_col in cat_cols:
    categorical_dims[cat_col] = 2

# Not used in forest cover
cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

# Not used in forest cover
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]


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
        
        # For forest cover, we pass the already one-hot encoded categorical parameters
        #  as numerical parameters, as this greatly increases accuracy and decreases
        #  fluctuations in val/test performance between epochs
        
        #cat_idxs=cat_idxs,
        #cat_dims=cat_dims,
        #cat_emb_dim=1,
        
        momentum=0.3,
        mask_type="entmax",
    )

    # Fit network on backdoored data
    clf.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
        eval_name=['train', 'valid'],
        max_epochs=EPOCHS, patience=EPOCHS,
        batch_size=1024, virtual_batch_size=128,
        #num_workers = 0,
    )
    
    # Evaluate backdoor    
    y_pred = clf.predict(X_test_backdoor.values)
    ASR = accuracy_score(y_pred=y_pred, y_true=y_test_backdoor.values)

    y_pred = clf.predict(X_test.values)
    BA = accuracy_score(y_pred=y_pred, y_true=y_test.values)
    
    return ASR, BA


# Start experiment
# Global results
ASR_results = []
BA_results = []

for poisoningRate in poisoningRates:
    # Run results
    ASR_run = []
    BA_run = []
    
    for run in range(RERUNS):
        ASR, BA = doExperiment(poisoningRate, backdoorFeatures, backdoorTriggerValues, targetLabel, run+1)
        print("Results for", poisoningRate, "Run", run+1)
        print("ASR:", ASR)
        print("BA:", BA)
        print("---------------------------------------")
        ASR_run.append(ASR)
        BA_run.append(BA)
        
    ASR_results.append(ASR_run)
    BA_results.append(BA_run)


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

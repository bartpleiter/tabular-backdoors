#!/bin/bash

# Make sure certain folders exist to prevent crashes
mkdir -p output
mkdir -p output/cleanlabel
mkdir -p FTtransformerCheckpoints
mkdir -p data/CLEAN-higgsFTT-1F-OOB
mkdir -p data/CLEAN-loanFTT-1F-OOB
mkdir -p data/CLEAN-covtypeFTT-1F-OOB
mkdir -p data/CLEAN-higgsFTT-3F-IB
mkdir -p data/CLEAN-loanFTT-3F-IB
mkdir -p data/CLEAN-covtypeFTT-3F-IB

# Run the experiment
python -m ExpCleanLabel.$1 > output/cleanlabel/$1.log

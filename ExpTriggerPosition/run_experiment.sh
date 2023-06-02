#!/bin/bash

# Make sure certain folders exist to prevent crashes
mkdir -p output
mkdir -p output/triggerposition
mkdir -p FTtransformerCheckpoints
mkdir -p data/covtypeFTT-FI
mkdir -p data/covtypeFTT-FI-num
mkdir -p data/loanFTT-FI
mkdir -p data/higgsFTT-FI
mkdir -p data/syn10FTT-FI

# Run the experiment
python -m ExpTriggerPosition.$1 > output/triggerposition/$1.log

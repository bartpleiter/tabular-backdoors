#!/bin/bash

# Make sure certain folders exist to prevent crashes
mkdir -p output
mkdir -p output/triggersize
mkdir -p FTtransformerCheckpoints
mkdir -p data/covtypeFTT-1F-OOB
mkdir -p data/covtypeFTT-2F-OOB
mkdir -p data/covtypeFTT-3F-OOB
mkdir -p data/loanFTT-1F-OOB
mkdir -p data/loanFTT-2F-OOB
mkdir -p data/loanFTT-3F-OOB
mkdir -p data/higgsFTT-1F-OOB
mkdir -p data/higgsFTT-2F-OOB
mkdir -p data/higgsFTT-3F-OOB

# Run the experiment
python -m ExpTriggerSize.$1 > output/triggersize/$1.log

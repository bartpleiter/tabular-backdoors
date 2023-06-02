#!/bin/bash

# Make sure certain folders exist to prevent crashes
mkdir -p output
mkdir -p output/inbounds
mkdir -p FTtransformerCheckpoints
mkdir -p data/covtypeFTT-3F-IB
mkdir -p data/loanFTT-3F-IB
mkdir -p data/higgsFTT-3F-IB

# Run the experiment
python -m ExpInBounds.$1 > output/inbounds/$1.log

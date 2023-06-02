# tabular-backdoors
Code repository for Master thesis on backdoor attacks on transformer-based DNNs for tabular data.

## Models used

- TabNet, https://arxiv.org/pdf/1908.07442.pdf, (used implementation from https://github.com/dreamquark-ai/tabnet)
- FT-Transformer, https://arxiv.org/pdf/2106.11959.pdf, (used implementation from https://github.com/Yura52/tabular-dl-revisiting-models)
- SAINT, https://arxiv.org/pdf/2106.01342.pdf, (used implementation from https://github.com/somepago/saint)

## Data used

- Forest Cover Type (CovType), http://archive.ics.uci.edu/ml/datasets/covertype
- Lending Club Loan (LOAN), https://www.kaggle.com/datasets/wordsforthewise/lending-club and https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv?select=LCDataDictionary.xlsx
- Higgs Boson (HIGGS), https://archive.ics.uci.edu/ml/datasets/HIGGS

## Overview
```text
tabular-backdoors           # Project directory
├── data                    # Contains datasets and preprocessing notebooks
├── ExpCleanLabel           # Experiment code for Clean Label Attack
├── ExpInBounds             # Experiment code for In Bounds Trigger
├── ExpTriggerPosition      # Experiment code for Trigger Position based on feature importance
├── ExpTriggerSize          # Experiment code for Trigger Size
├── SAINT                   # SAINT model code
├── FTtransformer           # FT-Transformer model code
└── Notebooks               # Other (smaller or parts of) experiments in the form of notebooks
    ├── FeatureImportances  # Notebooks to calculate feature importance scores and rankings
    └── Defences            # Notebooks on defences against our attacks
```

## Usage

### Install and enable environment

```bash
virtualenv tabularbackdoor
source tabularbackdoor/bin/activate
pip install -r requirements.txt

# To run the notebooks you also need:
pip install notebook
```

### Download and preprocess data

1. Download `accepted_2007_to_2018Q4.csv` from https://www.kaggle.com/datasets/wordsforthewise/lending-club and place in `data/LOAN/`
2. Download `LCDataDictionary.xlsx` from https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv?select=LCDataDictionary.xlsx and place in `data/LOAN/`
3. Download `HIGGS.csv.gz` from https://archive.ics.uci.edu/ml/datasets/HIGGS and extract `HIGGS.csv` to `data/HIGGS`
4. Run all four notebooks under `data/preprocess` to generate the `.pkl` files containing the datasets for the experiments

### Run main experiments

Run the shell script in any of the `Exp*` folders from the project root with the Python filename (without extension) as argument. Output will be logged to the output folder.

- NOTE: starting an experiment will override the previous log file of the same experiment.
- NOTE: depending on the machine, you might want to edit the GPU used to train each model. To do so, edit the `cuda:x` string (located somewhere on top) in each `.py` file.

Example:
```bash
bash ExpTriggerSize/run_experiment.sh TabNet_CovType_1F_OOB
```

To live view the log of a running experiment, use `tail -f` with the logfile as argument in a new terminal:

```bash
tail -f output/triggersize/TabNet_CovType_1F_OOB.log
```

### View results of main experiments

Output logs are found in the `output/` folder. All logs end with a section `EASY COPY PASTE RESULTS:` where you can copy the resulting lists containing the `ASR` and `BA` for each run.

### Run notebooks (e.g. Spectral Signatures defence)

See the `Notebooks/` folder for other (smaller or parts of) experiments in the form of notebooks. To run the defences, you must first run the appropiate `CreateModel` Notebook to create a backdoored model and dataset which can then be analyzed with the other Notebooks.

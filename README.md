# CGM accuracy (Bent et al. 2021 reproduction)

This repository contains the same pipeline as the Research_D'Amario project, stored under `InitialRuns/`. It reproduces Bent et al., *npj Digital Medicine* (2021): interstitial glucose classification and prediction from Empatica E4, Dexcom G6, food logs, and demographics.

The original module name `src.bent2021_pipeline` is **not** used here. Run the commands in this file, not those from the other project.

## Setup

From the repository root (`cgm_accuracy`):

```bash
conda create -n bent2021 python=3.10 pandas numpy scipy scikit-learn xgboost llvm-openmp -c conda-forge
conda activate bent2021
```

`llvm-openmp` is required by XGBoost on macOS.

Pip equivalent (OpenMP may still need a separate install on macOS):

```bash
pip install "pandas>=1.5" "numpy>=1.20" "scipy>=1.7" "scikit-learn>=1.0" "xgboost>=1.6"
```

## Data

Put patient folders in `InitialRuns/Data/` (currently empty except `.keep`):

```text
InitialRuns/Data/
  Demographics.csv
  015/
    ACC_015.csv
    BVP_015.csv
    Dexcom_015.csv
    EDA_015.csv
    Food_Log_015.csv
    HR_015.csv
    IBI_015.csv
    TEMP_015.csv
```

Additional patients use the same layout (`SENSOR_<id>.csv`, `Food_Log_<id>.csv`).

Always pass `--data-dir InitialRuns/Data`. If you omit it, the code looks for `Data/` at the **repository root**, which is the wrong folder.

## Commands

Run these from the **repository root**, with the conda environment activated.

Full pipeline (features + classification + regression):

```bash
python -m InitialRuns.src.cli --data-dir InitialRuns/Data --out-dir InitialRuns/outputs
```

Features only (no model training):

```bash
python -m InitialRuns.src.cli --data-dir InitialRuns/Data --out-dir InitialRuns/outputs --inspect-only --skip-bvp-inspect
```

One patient:

```bash
python -m InitialRuns.src.cli --data-dir InitialRuns/Data --out-dir InitialRuns/outputs --patients 015
```

| Flag | Meaning |
|---|---|
| `--data-dir InitialRuns/Data` | Patient folders (required for this repo layout) |
| `--out-dir InitialRuns/outputs` | Where reports and metrics are written |
| `--patients 015` | Process only the listed IDs |
| `--inspect-only` | Features and validation reports; skip training |
| `--skip-bvp-inspect` | Skip the large BVP file during validation |
| `--seed 42` | RNG seed (not reported in the paper; 42 is the default) |

With only one patient, feature engineering, classification, and the personalized 50/50 XGBoost model still run. Population leave-one-person-out CV is skipped until there are at least two patient folders.

ACC and BVP files are large. A full run can use several GB of RAM. Use `--skip-bvp-inspect` if you do not need the BVP validation summary; BVP is not a model feature.

## Outputs

Written to `InitialRuns/outputs/` (or whatever you pass to `--out-dir`):

| File | Contents |
|---|---|
| `validation_summary.txt` | File counts, time ranges, labels, missingness |
| `<id>_raw_inspect.json` | Per-file row counts, timestamps, sampling |
| `<id>_feature_inspect.json` | Feature dimensions, ranges, class counts |
| `<id>_features.csv` | One row per CGM sample with 69 features, glucose, and labels |
| `ALL_features.csv` | Concatenated feature table |
| `uncertainties.json` | Documented assumptions and missing paper details |
| `classification_metrics.json` | K-fold and 70/30 classification metrics |
| `regression_personalized_metrics.json` | Personalized XGBoost RMSE / MAPE / accuracy |
| `regression_population_metrics.json` | LOPOCV results, or a skip notice |
| `run_report.json` | Run configuration and high-level counts |

## Layout

```text
cgm_accuracy/
└── InitialRuns/
    ├── Data/                 # put patient folders here
    ├── outputs/             # created when you run
    └── src/                  # pipeline code
        ├── cli.py
        ├── pipeline.py
        ├── features.py
        └── ...
```

Ambiguities (food-log date shift, missing hyperparameters, and so on) are listed in `InitialRuns/src/uncertainties.py` and copied into `uncertainties.json` on each run.

"""Feature names and paper-traced constants.

Supplementary Table 1 lists 69 features. Names below follow that table,
with spaces in 'Biological Sex' replaced by an underscore for CSV headers.
"""

from __future__ import annotations

from typing import List

# Paper Methods: Dexcom G6 every 5 min; Empatica sampling rates.
CGM_INTERVAL = "5min"
EDA_FS_HZ = 4.0
TEMP_FS_HZ = 4.0
ACC_FS_HZ = 32.0
HR_FS_HZ = 1.0
BVP_FS_HZ = 64.0

# SI Table 1 / paper: EDA peaks, distance=1 s = 4 samples at 4 Hz, prominence 0.3 µS.
EDA_PEAK_DISTANCE_SAMPLES = 4
EDA_PEAK_PROMINENCE_US = 0.3
EDA_PEAK_HEIGHT = 0.0

# Paper: NN50 threshold 50 ms.
NN50_MS = 50.0

# Paper: recursive feature elimination keeps 20 features.
RFE_N_FEATURES = 20

# Paper: repeated stratified k-fold, 10 splits, 3 repeats.
KFOLD_SPLITS = 10
KFOLD_REPEATS = 3

# Paper: 70/30 train/test split.
TRAIN_TEST_FRACTION = 0.70

# Paper: RF feature selection, 1000 trees, importance cutoff 0.005.
RF_N_ESTIMATORS = 1000
RF_IMPORTANCE_CUTOFF = 0.005

# Paper: XGBoost max_depth=6, n_estimators=100, learning_rate=0.1.
XGB_MAX_DEPTH = 6
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = 0.1

# Assumed (U21): seed used wherever the paper is silent.
DEFAULT_RANDOM_STATE = 42

# SI Table 1 feature names (69).
FEATURE_NAMES: List[str] = [
    # Demographics / clinical / personalization
    "Biological_Sex",
    "HbA1c",
    "ID",
    # Data-driven EDA (5-min)
    "EDA_Mean",
    "EDA_Std",
    "EDA_Min",
    "EDA_Max",
    "EDA_Q1G",
    "EDA_Q3G",
    "EDA_Skew",
    # Data-driven HR
    "HR_Mean",
    "HR_Std",
    "HR_Min",
    "HR_Max",
    "HR_Q1G",
    "HR_Q3G",
    "HR_Skew",
    # Data-driven TEMP
    "TEMP_Mean",
    "TEMP_Std",
    "TEMP_Min",
    "TEMP_Max",
    "TEMP_Q1G",
    "TEMP_Q3G",
    "TEMP_Skew",
    # Data-driven ACC (vector magnitude)
    "ACC_Mean",
    "ACC_Std",
    "ACC_Min",
    "ACC_Max",
    "ACC_Q1G",
    "ACC_Q3G",
    "ACC_Skew",
    # Stress: EDA peaks
    "PeakEDA",
    "PeakEDA2hr_sum",
    "PeakEDA2hr_mean",
    # Stress: HRV
    "maxHRV",
    "minHRV",
    "medianHRV",
    "meanHRV",
    "SDNN",
    "NN50",
    "pNN50",
    "RMSSD",
    # Diet
    "calories2hr",
    "protein2hr",
    "sugar2hr",
    "carbs2hr",
    "calories8hr",
    "protein8hr",
    "sugar8hr",
    "carbs8hr",
    "calories24hr",
    "protein24hr",
    "sugar24hr",
    "carbs24hr",
    "Eat",
    "Eatcnt2hr",
    "Eatcnt8hr",
    "Eatcnt24hr",
    "Eatmean2hr",
    "Eatmean8hr",
    "Eatmean24hr",
    # Circadian
    "WakeTime",
    "Minfrommid",
    "Hourfrommid",
    # Activity
    "ACC_mean_2hrs",
    "ACC_max_2hrs",
    "Activity_bouts",
    "Activity24",
    "Activity1hr",
]

assert len(FEATURE_NAMES) == 69, len(FEATURE_NAMES)

# Features that should not be lagged by 5 min (U04).
# Clock features describe the time of the glucose sample, not a prior epoch.
NON_LAGGED_FEATURES = {"Biological_Sex", "HbA1c", "ID", "Minfrommid", "Hourfrommid"}

LABEL_ORDER = ["PersLow", "PersNorm", "PersHigh"]
LABEL_TO_INT = {"PersLow": 0, "PersNorm": 1, "PersHigh": 2}

SENSOR_FILE_STEMS = ["ACC", "BVP", "Dexcom", "EDA", "HR", "IBI", "TEMP"]

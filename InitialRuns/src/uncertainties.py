"""Catalog of ambiguities, missing details, and implementation choices.

Each item is tagged:
  EXPLICIT    — stated in the paper or Supplementary Information (SI)
  IMPLIED     — not word-for-word, but supported by surrounding methods/data
  ASSUMED     — required to run the code; not specified by the authors
  UNKNOWN     — cannot be determined from the available materials

These notes are written into the validation report so they remain auditable.
"""

from __future__ import annotations

from typing import Any, Dict, List


def catalog() -> List[Dict[str, Any]]:
    return [
        {
            "id": "U01_food_date_misalignment",
            "status": "ASSUMED",
            "what_is_ambiguous": (
                "Patient 015 food-log dates (2020-02-17 to 2020-02-25) do not "
                "overlap Dexcom (2020-07-19 to 2020-07-27) or Empatica "
                "(2020-07-05 to 2020-07-25). PhysioNet 1.1.3 notes that food-log "
                "dates were misaligned in earlier releases and were later "
                "shifted; the copy in Data/015 still has the February dates."
            ),
            "evidence": (
                "Paper Methods: food logs were recorded during the 8–10 day "
                "monitoring period. PhysioNet: data are date-shifted for "
                "de-identification; v1.1.3 'Updated misaligned food log dates'. "
                "PhysioNet 1.1.3 food log for 015 starts 2020-07-05, which "
                "overlaps Empatica but still does not overlap Dexcom."
            ),
            "interpretations": [
                "A: Leave food timestamps unchanged (food features will be zero "
                "during the CGM window).",
                "B: Shift food by an integer number of days so the first food "
                "calendar day equals the first CGM calendar day (maximizes "
                "overlap with the prediction target).",
                "C: Shift food to the first Empatica calendar day (PhysioNet "
                "1.1.3), which overlaps wearable cluster 1 (Jul 5–7) but not CGM.",
            ],
            "chosen": "B",
            "why": (
                "The paper requires simultaneous food, wearable, and CGM data. "
                "Glucose is the prediction target, so food is aligned to CGM. "
                "The shift is recorded in the validation report."
            ),
        },
        {
            "id": "U02_demographics_missing_from_data_folder",
            "status": "ASSUMED",
            "what_is_ambiguous": (
                "Biological sex and HbA1c are two of the 69 features (SI Table 1) "
                "but are not present in Data/015."
            ),
            "evidence": (
                "PhysioNet BIG IDEAs Lab dataset ships Demographics.csv; ID 15 "
                "is FEMALE, HbA1c 5.5. Paper SI Table 2 reports cohort sex split "
                "but not per-participant values."
            ),
            "interpretations": [
                "A: Leave Biological_Sex and HbA1c as missing.",
                "B: Use the PhysioNet Demographics.csv values for this cohort.",
            ],
            "chosen": "B",
            "why": (
                "These values are the published companion demographics for the "
                "same patient IDs. They are stored in Data/Demographics.csv and "
                "are not inferred from the Dexcom header (DOB 1/1/1970 is "
                "de-identified filler)."
            ),
        },
        {
            "id": "U03_sex_encoding",
            "status": "ASSUMED",
            "what_is_ambiguous": "How biological sex is encoded as a model input.",
            "evidence": "SI Table 1 lists 'Biological Sex' with calculation N/A.",
            "interpretations": ["Female=0, Male=1", "Female=1, Male=0", "one-hot"],
            "chosen": "Female=0, Male=1",
            "why": (
                "Unspecified in the paper. This encoding matches the authors' "
                "later public refactor (brinnaebent/glucose-prediction). Tree "
                "models are invariant to 0/1 polarity."
            ),
        },
        {
            "id": "U04_historical_lag",
            "status": "IMPLIED",
            "what_is_ambiguous": (
                "How 'historical (5 min to 24 h prior)' is applied to features "
                "computed on 5-min epochs that are simultaneous with CGM."
            ),
            "evidence": (
                "Paper Methods, Feature engineering: 'All features used in "
                "modeling were historical (5 min to 24 h prior to the "
                "measurement being predicted).'"
            ),
            "interpretations": [
                "A: Compute 5-min epoch features, then shift all non-demographic "
                "features by one 5-min bin before pairing with glucose at time t.",
                "B: Compute features on [t-5min, t) and treat that as historical.",
                "C: No extra lag; rolling windows already look backward.",
            ],
            "chosen": "A",
            "why": (
                "Option A is the strict reading of '5 min to 24 h prior' and "
                "prevents using the wearable epoch concurrent with the glucose "
                "value being predicted."
            ),
        },
        {
            "id": "U05_glucose_grid",
            "status": "IMPLIED",
            "what_is_ambiguous": (
                "Whether glucose is kept at native Dexcom timestamps or "
                "resampled onto a round 5-min clock, including interpolation."
            ),
            "evidence": (
                "Paper: Dexcom G6 records every 5 min; models predict at 5-min "
                "intervals. No interpolation procedure is described."
            ),
            "interpretations": [
                "A: Keep native EGV timestamps; floor each to a 5-min bin.",
                "B: Resample/interpolate glucose onto an exact 5-min grid.",
            ],
            "chosen": "A",
            "why": (
                "Interpolation is not described. Native EGV values are the "
                "ground truth. Each EGV is mapped to floor(timestamp, 5min)."
            ),
        },
        {
            "id": "U06_sensor_std_and_skew",
            "status": "EXPLICIT",
            "what_is_ambiguous": (
                "SI gives population std (divide by N) and a specific skew "
                "formula. pandas defaults differ."
            ),
            "evidence": "SI Table 1 formulas for *_Std and *_Skew.",
            "interpretations": ["Follow SI formulas", "Use pandas sample std / G1 skew"],
            "chosen": "Follow SI formulas",
            "why": "SI is the primary authority for feature calculation.",
        },
        {
            "id": "U07_quartile_interpolation",
            "status": "ASSUMED",
            "what_is_ambiguous": "How Q1/Q3 are interpolated for finite samples.",
            "evidence": "SI: 'first quartile' / 'third quartile' with no method.",
            "interpretations": ["linear (numpy default)", "lower/higher/midpoint"],
            "chosen": "linear (numpy.quantile method='linear')",
            "why": "Not specified; numpy's default linear interpolation is used.",
        },
        {
            "id": "U08_eda_peak_scope",
            "status": "IMPLIED",
            "what_is_ambiguous": (
                "Whether find_peaks is run on the entire EDA recording or "
                "within each 5-min window / contiguous session."
            ),
            "evidence": (
                "Paper: SciPy find_peaks, distance=1 s (4 samples), prominence "
                "0.3 µS; count peaks per 5-min interval. SI also lists height=0."
            ),
            "interpretations": [
                "A: Global find_peaks on the concatenated array.",
                "B: find_peaks within contiguous sessions (break on time gaps), "
                "then count per 5-min bin.",
                "C: find_peaks independently inside each 5-min window.",
            ],
            "chosen": "B",
            "why": (
                "Patient 015 Empatica files concatenate disjoint sessions "
                "(Jul 5–7 then Jul 19–25). Treating those as adjacent samples "
                "would invent peaks at session boundaries. height=0, distance=4, "
                "prominence=0.3 follow SI."
            ),
        },
        {
            "id": "U09_hrv_units_and_formulas",
            "status": "ASSUMED",
            "what_is_ambiguous": (
                "IBI files store seconds. SDNN/RMSSD/pNN50 formulas differ "
                "slightly between the paper text, SI, and standard HRV definitions."
            ),
            "evidence": (
                "Paper: 8 HRV metrics over each 5-min interval; NN50 uses 50 ms. "
                "SI: SDNN = sqrt(σ²) of IBI; pNN50 = NN50/len(N); RMSSD divides "
                "by len(N). Standard RMSSD divides by number of successive diffs."
            ),
            "interpretations": [
                "Convert IBI to ms; SDNN sample std (ddof=1); RMSSD = rms of "
                "successive diffs (divide by n_diffs); pNN50 = NN50 / n_IBI.",
                "Use SI RMSSD denominator len(N) and/or keep IBI in seconds.",
            ],
            "chosen": (
                "IBI converted to ms. SDNN: sample std. RMSSD: rms of successive "
                "differences (paper wording). pNN50: NN50 / n_IBI (SI)."
            ),
            "why": (
                "50 ms threshold requires milliseconds. Paper wording for RMSSD "
                "matches the standard definition more closely than the SI "
                "denominator. pNN50 follows SI's len(N)."
            ),
        },
        {
            "id": "U10_activity_historical_mean",
            "status": "IMPLIED",
            "what_is_ambiguous": (
                "What 'average of the prior historical data from the individual' "
                "means, including min_periods."
            ),
            "evidence": (
                "Paper: compare 5-min mean ACC and HR to the average of prior "
                "historical data; both must be above that average to mark an "
                "activity bout."
            ),
            "interpretations": [
                "Expanding mean of all prior 5-min epochs (shifted by 1).",
                "Prior 24 h mean only.",
            ],
            "chosen": "Expanding mean of all prior epochs, min_periods=1, shifted by 1.",
            "why": "Matches 'prior historical data from the individual'.",
        },
        {
            "id": "U11_wake_algorithm",
            "status": "ASSUMED",
            "what_is_ambiguous": (
                "WakeTime detection is underspecified. Paper and SI disagree on "
                "the polarity of the 0/1 assignment. Search window, slope "
                "threshold, and 'consistently higher' are not fully defined."
            ),
            "evidence": (
                "Paper: if two of four (ACC/HR mean/std) are below the day "
                "average, assign 0; else 1; average over 3 h; wake when slope "
                "sharply changes and remains higher 25 and 75 min later. "
                "SI: 'If points >2, binary assignment of 1' (opposite polarity)."
            ),
            "interpretations": [
                "Follow paper polarity (>=2 of 4 below daily mean → 0).",
                "Follow SI polarity (points>2 → 1).",
            ],
            "chosen": (
                "Paper polarity: >=2 of 4 below that day's mean → 0, else 1. "
                "3 h rolling mean. WakeTime = first time in 04:00–14:00 where "
                "the 3 h mean is higher 25 min and 75 min later and the local "
                "slope is positive. WakeTime in minutes after midnight. "
                "Search window 04:00–14:00 is not in the paper."
            ),
            "why": (
                "Paper text is the primary authority for the 0/1 rule. The "
                "morning search window is an assumption needed to avoid "
                "night-time false detections; it is flagged."
            ),
        },
        {
            "id": "U12_eat_flag",
            "status": "ASSUMED",
            "what_is_ambiguous": (
                "Whether Eat=1 for any logged item or only caloric items, and "
                "whether eating spans time_begin–time_end."
            ),
            "evidence": (
                "Paper: unique meal, snack, or caloric beverage → binary 1 on "
                "that interval. SI: '1 if currently eating'."
            ),
            "interpretations": [
                "Mark the 5-min bin containing time_begin for every log row.",
                "Only rows with calorie>0.",
                "Fill 1 from time_begin through time_end.",
            ],
            "chosen": (
                "Eat=1 on the 5-min bin of time_begin for every food-log row "
                "(including 0-calorie items). Duration/time_end is not filled. "
                "Multiple items at the same timestamp are one Eat event; "
                "nutrients are summed."
            ),
            "why": (
                "Paper assigns 'that interval' at consumption time. Duration "
                "filling is not described. Zero-calorie logged items are still "
                "consumption events."
            ),
        },
        {
            "id": "U13_pers_labels_min_history",
            "status": "IMPLIED",
            "what_is_ambiguous": (
                "Whether PersHigh/PersLow/PersNorm are assigned before 24 h of "
                "CGM history exists."
            ),
            "evidence": (
                "Paper: exceed / fall below / stay within 1 SD of the mean for "
                "the last 24 h."
            ),
            "interpretations": [
                "Require a full 24 h lookback before labeling.",
                "Use expanding mean/std until 24 h is available.",
            ],
            "chosen": "Require at least 24 h of elapsed CGM lookback to assign a label.",
            "why": "Matches the stated 24 h personalized mean.",
        },
        {
            "id": "U14_class_balancing",
            "status": "ASSUMED",
            "what_is_ambiguous": (
                "How classes were balanced to N=8666. Method (under/over-sample) "
                "and per-class target size are not stated."
            ),
            "evidence": (
                "Paper: classes balanced to N=8666 because the full ~25,000 "
                "points were majority PersNorm."
            ),
            "interpretations": [
                "Undersample each class to the minority class size.",
                "Undersample majority classes to a fixed count summing to 8666.",
            ],
            "chosen": (
                "Random undersample without replacement so all three classes "
                "equal the minority-class count. N=8666 is a 16-patient figure "
                "and cannot be reproduced from patient 015 alone."
            ),
            "why": "Most direct reading of 'balanced classes'. Seeded RNG.",
        },
        {
            "id": "U15_missing_feature_values",
            "status": "ASSUMED",
            "what_is_ambiguous": (
                "No missing-data procedure is described. Patient 015 has CGM "
                "days with no Empatica data (e.g. Jul 21, 23, 26, 27)."
            ),
            "evidence": "Paper does not mention imputation or row dropping.",
            "interpretations": [
                "Keep all CGM rows; leave wearable features as NaN.",
                "Drop rows with any NaN feature.",
                "Impute.",
            ],
            "chosen": (
                "Keep all labeled CGM rows. XGBoost uses native NaN handling. "
                "sklearn DecisionTree/LogisticRegression/RFE/RandomForest get "
                "median imputation fit on the training fold only."
            ),
            "why": (
                "Dropping would discard CGM days the paper would still have "
                "labeled. Imputation is required for sklearn and is flagged."
            ),
        },
        {
            "id": "U16_dt_lr_hyperparameters",
            "status": "UNKNOWN",
            "what_is_ambiguous": (
                "Decision tree and logistic regression hyperparameters "
                "(max_depth, min_samples_split, C, class encoding for R², "
                "whether the 70/30 split is stratified) are not given."
            ),
            "evidence": "Paper reports metrics only; sklearn is cited.",
            "interpretations": ["sklearn defaults", "tuned but unreported values"],
            "chosen": (
                "sklearn defaults. 70/30 split is stratified. R² uses integer "
                "codes PersLow=0, PersNorm=1, PersHigh=2, with negative R² "
                "clipped to 0 as described. random_state=42 (not in the paper)."
            ),
            "why": "No values are available; defaults plus an explicit seed.",
        },
        {
            "id": "U17_xgb_unspecified_params",
            "status": "ASSUMED",
            "what_is_ambiguous": (
                "Only max_depth=6, n_estimators=100, learning_rate=0.1 are "
                "stated. subsample, objective, tree_method, random_state are not."
            ),
            "evidence": "Paper Glucose prediction paragraph.",
            "interpretations": ["XGBoost library defaults for unspecified params"],
            "chosen": (
                "Set only the three stated hyperparameters plus random_state=42. "
                "Other XGBoost defaults remain."
            ),
            "why": "Do not add undocumented regularization or hist tree method.",
        },
        {
            "id": "U18_rf_feature_selection_imputation",
            "status": "ASSUMED",
            "what_is_ambiguous": (
                "Random forest (1000 trees, importance cutoff 0.005) cannot "
                "fit on NaNs."
            ),
            "evidence": "Paper: RF impurity importance, cutoff=0.005, 1000 trees.",
            "interpretations": ["Median impute train fold", "Drop NaN rows for RF"],
            "chosen": "Median impute using the training fold only, then fit RF.",
            "why": "Needed to run the stated RF selector; not described in the paper.",
        },
        {
            "id": "U19_rfe_estimator_details",
            "status": "ASSUMED",
            "what_is_ambiguous": (
                "RFE selects 20 features; they chose the decision-tree estimator "
                "after comparing several methods. RFE step size and DT params "
                "are not given."
            ),
            "evidence": "Paper Classification of glucose excursions.",
            "interpretations": ["RFE(DecisionTreeClassifier, n_features_to_select=20)"],
            "chosen": "sklearn RFE with DecisionTreeClassifier defaults, 20 features.",
            "why": "Matches the stated estimator and count; other details unknown.",
        },
        {
            "id": "U20_bvp_unused",
            "status": "EXPLICIT",
            "what_is_ambiguous": "BVP is collected but is not one of the 69 features.",
            "evidence": (
                "Paper uses IBI derived from PPG for HRV; SI Table 1 has no BVP "
                "summary statistics."
            ),
            "interpretations": ["Do not use BVP as a model feature."],
            "chosen": "Inspect BVP for validation only; do not engineer BVP features.",
            "why": "Not in the 69-feature list.",
        },
        {
            "id": "U21_random_seed",
            "status": "UNKNOWN",
            "what_is_ambiguous": "No random seed is reported.",
            "evidence": "Paper Methods.",
            "interpretations": ["Unset (non-reproducible)", "Any fixed seed"],
            "chosen": "random_state=42 everywhere a seed is required, configurable.",
            "why": "Required for deterministic undersampling, RFE, RF, XGBoost, splits.",
        },
        {
            "id": "U22_acc_vector_magnitude_units",
            "status": "IMPLIED",
            "what_is_ambiguous": (
                "Empatica ACC is typically in units of 1/64 g. The paper uses "
                "vector magnitude of the three axes and does not convert units."
            ),
            "evidence": "Paper Feature engineering; SI ACC_* formulas on x_i.",
            "interpretations": [
                "Magnitude of raw x,y,z as stored.",
                "Convert to g then take magnitude.",
            ],
            "chosen": "sqrt(x^2+y^2+z^2) on the stored integer/float values.",
            "why": "No unit conversion is described.",
        },
        {
            "id": "U23_lopo_single_patient",
            "status": "EXPLICIT",
            "what_is_ambiguous": (
                "LOPOCV is defined over 16 participants. Only patient 015 is present."
            ),
            "evidence": "Paper Glucose prediction: iterate over each participant as test.",
            "interpretations": ["Skip LOPOCV until >=2 patient folders exist."],
            "chosen": "Implement LOPOCV; skip at runtime if fewer than 2 patients.",
            "why": "Cannot leave one person out of a one-person dataset.",
        },
        {
            "id": "U24_author_refactor_10min_window",
            "status": "EXPLICIT",
            "what_is_ambiguous": (
                "The authors' 2025 public refactor uses a 10-min rolling window "
                "for sensor summary statistics. The paper/SI say 5-min intervals."
            ),
            "evidence": "SI: N = number of points in a 5-minute interval.",
            "interpretations": ["5-min (paper/SI)", "10-min (later refactor)"],
            "chosen": "5-min intervals as in the paper and SI.",
            "why": "Paper/SI are the primary authority.",
        },
    ]


def as_jsonable() -> List[Dict[str, Any]]:
    return catalog()

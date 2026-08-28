"""Assemble the 69 features on the 5-min CGM grid (paper Methods; SI Table 1)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .activity_circadian import activity_features, circadian_clock_features, wake_time_feature
from .constants import FEATURE_NAMES, NON_LAGGED_FEATURES
from .food import food_features
from .io import PatientData
from .labels import personalized_labels
from .sensor_stats import five_min_stats
from .stress import eda_peak_features, hrv_features


def build_feature_table(patient: PatientData) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return one row per CGM sample with 69 features, glucose, and labels."""
    notes: Dict[str, Any] = {"patient_id": patient.patient_id, "discarded": {}}

    eda_s = five_min_stats(patient.eda["EDA"] if not patient.eda.empty else pd.Series(dtype=float), "EDA")
    temp_s = five_min_stats(patient.temp["TEMP"] if not patient.temp.empty else pd.Series(dtype=float), "TEMP")
    hr_s = five_min_stats(patient.hr["HR"] if not patient.hr.empty else pd.Series(dtype=float), "HR")
    acc_s = five_min_stats(patient.acc["ACC"] if not patient.acc.empty else pd.Series(dtype=float), "ACC")
    peaks = eda_peak_features(patient.eda)
    hrv = hrv_features(patient.ibi)

    parts = [p for p in (eda_s, temp_s, hr_s, acc_s, peaks, hrv) if p is not None and not p.empty]
    if not parts:
        raise RuntimeError(f"No wearable features could be computed for {patient.patient_id}")
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.join(p, how="outer")
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.index.name = "Time"

    merged = activity_features(merged)
    merged = merged.join(circadian_clock_features(merged.index))
    merged["WakeTime"] = wake_time_feature(merged)

    food_f = food_features(patient.food, merged.index)
    if not food_f.empty:
        merged = merged.join(food_f, how="left")

    sex = patient.demographics.get("Biological_Sex", np.nan)
    hba1c = patient.demographics.get("HbA1c", np.nan)
    merged["Biological_Sex"] = sex
    merged["HbA1c"] = hba1c
    try:
        merged["ID"] = int(str(patient.patient_id).lstrip("0") or "0")
    except ValueError:
        merged["ID"] = pd.factorize([patient.patient_id])[0][0]

    # Paper: features are historical (5 min to 24 h prior). U04: lag non-demographic
    # features by one 5-min bin so the concurrent epoch is not used.
    lag_cols = [c for c in merged.columns if c not in NON_LAGGED_FEATURES]
    shifted = merged[lag_cols].shift(1, freq="5min")
    lagged = merged.copy()
    lagged[lag_cols] = shifted.reindex(merged.index)

    # Map each native CGM timestamp to floor(t, 5min) and attach lagged features.
    labels = personalized_labels(patient.glucose["Glucose"])
    labels = labels.copy()
    labels["bin"] = labels.index.floor("5min")
    feat = lagged.reindex(labels["bin"].to_numpy())
    feat.index = labels.index
    # Clock features describe the glucose sample time, not the lagged bin.
    clock = circadian_clock_features(feat.index)
    feat["Minfrommid"] = clock["Minfrommid"].to_numpy()
    feat["Hourfrommid"] = clock["Hourfrommid"].to_numpy()

    table = feat.join(labels[["Glucose", "glucose_mean_24h", "glucose_std_24h", "label", "label_int", "label_ready"]])
    table["patient_id"] = patient.patient_id
    table.index.name = "Time"

    # Ensure all 69 feature columns exist.
    for c in FEATURE_NAMES:
        if c not in table.columns:
            table[c] = np.nan

    n_glu = int(table["Glucose"].notna().sum())
    n_labeled = int(table["label"].notna().sum())
    n_unlabeled_warmup = int((~table["label_ready"]).sum())
    notes["n_glucose_rows"] = n_glu
    notes["n_labeled"] = n_labeled
    notes["n_unlabeled_insufficient_24h"] = n_unlabeled_warmup
    notes["label_counts"] = table["label"].value_counts(dropna=False).to_dict()
    notes["feature_missing_frac"] = {
        c: float(table[c].isna().mean()) for c in FEATURE_NAMES
    }
    notes["n_rows_any_feature_nan"] = int(table[FEATURE_NAMES].isna().any(axis=1).sum())
    notes["historical_lag"] = "shift non-demographic features by 5 min (U04)"
    notes["food_alignment"] = patient.food_alignment
    notes["demographics"] = {
        "found": patient.demographics.get("found"),
        "Gender_raw": patient.demographics.get("Gender_raw"),
        "Biological_Sex": patient.demographics.get("Biological_Sex"),
        "HbA1c": patient.demographics.get("HbA1c"),
    }
    return table, notes

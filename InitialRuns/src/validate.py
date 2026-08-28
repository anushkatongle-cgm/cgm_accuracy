"""Pre-training validation / debugging reports (task section 7)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .constants import FEATURE_NAMES
from .io import PatientData
from .uncertainties import catalog


def _series_report(s: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(s, errors="coerce")
    return {
        "n": int(s.shape[0]),
        "n_missing": int(s.isna().sum()),
        "min": float(s.min()) if s.notna().any() else None,
        "max": float(s.max()) if s.notna().any() else None,
        "mean": float(s.mean()) if s.notna().any() else None,
        "std": float(s.std()) if s.notna().any() else None,
        "p25": float(s.quantile(0.25)) if s.notna().any() else None,
        "p50": float(s.quantile(0.50)) if s.notna().any() else None,
        "p75": float(s.quantile(0.75)) if s.notna().any() else None,
    }


def patient_raw_report(patient: PatientData) -> Dict[str, Any]:
    return {
        "patient_id": patient.patient_id,
        "folder": str(patient.folder),
        "files": patient.file_reports,
        "food_alignment": patient.food_alignment,
        "demographics": patient.demographics,
        "bvp": patient.bvp_inspect,
        "glucose_n": int(len(patient.glucose)),
        "glucose_range": [
            str(patient.glucose.index.min()) if len(patient.glucose) else None,
            str(patient.glucose.index.max()) if len(patient.glucose) else None,
        ],
        "eda_n": int(len(patient.eda)),
        "temp_n": int(len(patient.temp)),
        "hr_n": int(len(patient.hr)),
        "acc_n": int(len(patient.acc)),
        "ibi_n": int(len(patient.ibi)),
        "food_n": int(len(patient.food)),
    }


def feature_table_report(table: pd.DataFrame, feature_notes: Dict[str, Any]) -> Dict[str, Any]:
    labeled = table.dropna(subset=["label"])
    return {
        "n_rows": int(len(table)),
        "n_glucose": int(table["Glucose"].notna().sum()),
        "n_labeled": int(len(labeled)),
        "time_min": str(table.index.min()) if len(table) else None,
        "time_max": str(table.index.max()) if len(table) else None,
        "label_counts": {str(k): int(v) for k, v in table["label"].value_counts(dropna=False).items()},
        "feature_count_expected": 69,
        "feature_count_present": int(sum(c in table.columns for c in FEATURE_NAMES)),
        "missing_features": [c for c in FEATURE_NAMES if c not in table.columns],
        "input_shape_if_all_69": [int(len(table)), 69],
        "input_shape_labeled": [int(len(labeled)), 69],
        "feature_ranges": {c: _series_report(table[c]) for c in FEATURE_NAMES if c in table.columns},
        "glucose": _series_report(table["Glucose"]),
        "feature_notes": feature_notes,
        "n_discarded_unlabeled_24h": int((~table["label_ready"]).sum()) if "label_ready" in table.columns else None,
    }


def write_text_summary(path, reports: List[Dict[str, Any]], feature_reports: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append("Bent et al. 2021 pipeline — validation summary")
    lines.append("=" * 72)
    for r, fr in zip(reports, feature_reports):
        lines.append(f"\nPatient {r['patient_id']}")
        lines.append("-" * 40)
        files = r.get("files", {})
        lines.append(f"  Files present (approx): {files.get('n_files_present')}")
        lines.append(f"  Dexcom EGV: {r['glucose_n']}  range {r['glucose_range']}")
        lines.append(f"  EDA rows: {r['eda_n']}  TEMP: {r['temp_n']}  HR: {r['hr_n']}")
        lines.append(f"  ACC rows: {r['acc_n']}  IBI: {r['ibi_n']}  Food rows: {r['food_n']}")
        fa = r.get("food_alignment", {})
        lines.append(f"  Food alignment: {fa.get('method')} shift_days={fa.get('shift_days')}")
        lines.append(f"  {fa.get('note')}")
        demo = r.get("demographics", {})
        lines.append(
            f"  Demographics found={demo.get('found')} "
            f"Gender={demo.get('Gender_raw')} Sex_enc={demo.get('Biological_Sex')} "
            f"HbA1c={demo.get('HbA1c')}"
        )
        lines.append(f"  Feature table rows: {fr['n_rows']}  labeled: {fr['n_labeled']}")
        lines.append(f"  Labels: {fr['label_counts']}")
        lines.append(f"  Expected model input shape (labeled): {fr['input_shape_labeled']}")
        miss = fr["feature_notes"].get("feature_missing_frac", {})
        worst = sorted(miss.items(), key=lambda kv: -kv[1])[:8]
        lines.append("  Highest feature missingness:")
        for name, frac in worst:
            lines.append(f"    {name}: {frac:.3f}")
    lines.append("\nUncertainties (see uncertainties.py / validation JSON)")
    lines.append("-" * 40)
    for u in catalog():
        lines.append(f"  [{u['status']}] {u['id']}: {u['what_is_ambiguous'][:120]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

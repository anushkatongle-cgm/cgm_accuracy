"""End-to-end reproduction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from .classify import run_classification
from .constants import DEFAULT_RANDOM_STATE, FEATURE_NAMES
from .features import build_feature_table
from .io import (
    default_data_dir,
    discover_patient_ids,
    dump_json,
    json_default,
    load_patient,
    project_root,
)
from .regress import personalized_xgb, population_lopocv
from .uncertainties import catalog
from .validate import feature_table_report, patient_raw_report, write_text_summary


def run_pipeline(
    data_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    patient_ids: Optional[Sequence[str]] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    inspect_only: bool = False,
    skip_models: bool = False,
    inspect_bvp: bool = True,
) -> Dict[str, Any]:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    out_dir = Path(out_dir) if out_dir else project_root() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = list(patient_ids) if patient_ids else discover_patient_ids(data_dir)
    if not ids:
        raise FileNotFoundError(f"No patient folders found in {data_dir}")

    raw_reports = []
    feat_notes = []
    feat_reports = []
    tables = []

    for pid in ids:
        patient = load_patient(data_dir, pid, inspect_bvp=inspect_bvp)
        raw = patient_raw_report(patient)
        raw_reports.append(raw)
        dump_json(out_dir / f"{patient.patient_id}_raw_inspect.json", raw)

        table, notes = build_feature_table(patient)
        feat_notes.append(notes)
        fr = feature_table_report(table, notes)
        feat_reports.append(fr)
        dump_json(out_dir / f"{patient.patient_id}_feature_inspect.json", fr)

        table_out = table.reset_index()
        table_out.to_csv(out_dir / f"{patient.patient_id}_features.csv", index=False)
        tables.append(table)
        print(
            f"[{patient.patient_id}] glucose={notes['n_glucose_rows']} "
            f"labeled={notes['n_labeled']} labels={notes['label_counts']}"
        )

    all_table = pd.concat(tables, axis=0).sort_index()
    all_csv = out_dir / "ALL_features.csv"
    all_table.reset_index().to_csv(all_csv, index=False)

    write_text_summary(out_dir / "validation_summary.txt", raw_reports, feat_reports)
    dump_json(out_dir / "uncertainties.json", catalog())

    model_results: Dict[str, Any] = {}
    if not inspect_only and not skip_models:
        print("Running classification models...")
        clf = run_classification(all_table, random_state=random_state)
        dump_json(out_dir / "classification_metrics.json", clf)
        model_results["classification"] = clf

        print("Running personalized XGBoost (50/50)...")
        pers = personalized_xgb(all_table, random_state=random_state)
        pers_save = {k: v for k, v in pers.items() if k not in ("predictions", "importances")}
        dump_json(out_dir / "regression_personalized_metrics.json", pers_save)
        if not pers["predictions"].empty:
            pers["predictions"].to_csv(out_dir / "regression_personalized_preds.csv", index=False)
        if not pers["importances"].empty:
            pers["importances"].to_csv(out_dir / "regression_personalized_rf_importances.csv", index=False)
        model_results["personalized"] = pers_save

        print("Running population LOPOCV XGBoost...")
        pop = population_lopocv(all_table, random_state=random_state)
        pop_save = {k: v for k, v in pop.items() if k not in ("predictions", "importances")}
        dump_json(out_dir / "regression_population_metrics.json", pop_save)
        if isinstance(pop.get("predictions"), pd.DataFrame) and not pop["predictions"].empty:
            pop["predictions"].to_csv(out_dir / "regression_population_preds.csv", index=False)
        if isinstance(pop.get("importances"), pd.DataFrame) and not pop["importances"].empty:
            pop["importances"].to_csv(out_dir / "regression_population_rf_importances.csv", index=False)
        model_results["population_lopocv"] = pop_save

    run_report = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "patient_ids": ids,
        "n_patients": len(ids),
        "n_feature_rows": int(len(all_table)),
        "n_features": 69,
        "feature_names": FEATURE_NAMES,
        "random_state": random_state,
        "inspect_only": inspect_only,
        "models": model_results,
    }
    dump_json(out_dir / "run_report.json", run_report)
    print(f"Wrote outputs to {out_dir}")
    return run_report

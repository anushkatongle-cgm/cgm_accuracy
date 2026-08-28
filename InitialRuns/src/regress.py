"""Interstitial glucose regression (paper: Glucose prediction).

Population model: XGBoost + per-fold RF feature selection, LOPOCV.
Personalized model: first contiguous 50% train, remaining 50% test, per patient.
Naive baselines: train-set mean and median.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

from .constants import (
    DEFAULT_RANDOM_STATE,
    FEATURE_NAMES,
    RF_IMPORTANCE_CUTOFF,
    RF_N_ESTIMATORS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def accuracy_from_mape(m: float) -> float:
    # Paper: accuracy = 100 - MAPE.
    return float(100.0 - m)


def _xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_NAMES].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = df["Glucose"].astype(float)
    return X, y


def _rf_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Tuple[List[str], pd.Series, SimpleImputer]:
    """RF impurity importance, keep features with importance >= 0.005.

    U18: median impute on the training fold only (not described in the paper).
    """
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train)
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_imp, y_train.to_numpy())
    imp = pd.Series(rf.feature_importances_, index=X_train.columns, name="importance")
    keep = imp[imp >= RF_IMPORTANCE_CUTOFF].index.tolist()
    if not keep:
        # Fallback: keep the single most important feature so training can proceed.
        keep = [imp.sort_values(ascending=False).index[0]]
    return keep, imp.sort_values(ascending=False), imputer


def _xgb() -> XGBRegressor:
    # U17: only the three paper hyperparameters plus an explicit seed.
    return XGBRegressor(
        max_depth=XGB_MAX_DEPTH,
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=XGB_LEARNING_RATE,
        random_state=DEFAULT_RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def _eval(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m = mape(y_true, y_pred)
    return {"rmse": rmse(y_true, y_pred), "mape": m, "accuracy": accuracy_from_mape(m)}


def naive_baselines(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    mean_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
    median_pred = np.full_like(y_test, fill_value=np.median(y_train), dtype=float)
    return {
        "mean": _eval(y_test, mean_pred),
        "median": _eval(y_test, median_pred),
    }


def personalized_xgb(
    table: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, Any]:
    """Train on the first contiguous 50% of each participant; test on the rest."""
    results = []
    importances = []
    preds_frames = []
    for pid, g in table.groupby("patient_id", sort=True):
        g = g.sort_index()
        g = g[g["Glucose"].notna()]
        n = len(g)
        if n < 20:
            results.append({"patient_id": pid, "skipped": True, "reason": f"n={n}<20"})
            continue
        split = n // 2
        train_p, test_p = g.iloc[:split], g.iloc[split:]
        Xtr, ytr = _xy(train_p)
        Xte, yte = _xy(test_p)
        keep, imp, _ = _rf_select(Xtr, ytr, random_state)
        importances.append(imp.rename(pid).to_frame("importance").assign(patient_id=pid, feature=imp.index))
        model = _xgb()
        # XGBoost handles NaN natively; do not impute for the booster (U15).
        model.fit(Xtr[keep], ytr)
        yhat = model.predict(Xte[keep])
        metrics = _eval(yte.to_numpy(), yhat)
        naive = naive_baselines(ytr.to_numpy(), yte.to_numpy())
        results.append({
            "patient_id": pid,
            "skipped": False,
            "n_train": int(len(train_p)),
            "n_test": int(len(test_p)),
            "n_features": int(len(keep)),
            "selected_features": keep,
            **metrics,
            "naive": naive,
        })
        preds_frames.append(pd.DataFrame({
            "Time": test_p.index,
            "patient_id": pid,
            "y_true": yte.to_numpy(),
            "y_pred": yhat,
        }))
    metrics_df = pd.DataFrame([r for r in results if not r.get("skipped")])
    summary = {}
    if not metrics_df.empty:
        for c in ("rmse", "mape", "accuracy"):
            summary[c] = {"mean": float(metrics_df[c].mean()), "std": float(metrics_df[c].std(ddof=0) if len(metrics_df) > 1 else 0.0)}
    return {
        "per_patient": results,
        "summary": summary,
        "predictions": pd.concat(preds_frames, ignore_index=True) if preds_frames else pd.DataFrame(),
        "importances": pd.concat(importances, ignore_index=True) if importances else pd.DataFrame(),
    }


def population_lopocv(
    table: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, Any]:
    ids = sorted(table["patient_id"].astype(str).unique())
    if len(ids) < 2:
        return {
            "skipped": True,
            "reason": (
                "LOPOCV requires >=2 participants (paper: 16). "
                f"Found {len(ids)}: {ids}"
            ),
            "n_participants": len(ids),
        }
    fold_rows = []
    importances = []
    preds_frames = []
    usable = table[table["Glucose"].notna()].copy()
    for pid in ids:
        train_df = usable[usable["patient_id"].astype(str) != pid]
        test_df = usable[usable["patient_id"].astype(str) == pid]
        if train_df.empty or test_df.empty:
            fold_rows.append({"patient_id": pid, "skipped": True, "reason": "empty fold"})
            continue
        Xtr, ytr = _xy(train_df)
        Xte, yte = _xy(test_df)
        keep, imp, _ = _rf_select(Xtr, ytr, random_state)
        importances.append(imp.to_frame("importance").assign(fold_id=pid, feature=imp.index))
        model = _xgb()
        model.fit(Xtr[keep], ytr)
        yhat = model.predict(Xte[keep])
        metrics = _eval(yte.to_numpy(), yhat)
        naive = naive_baselines(ytr.to_numpy(), yte.to_numpy())
        fold_rows.append({
            "patient_id": pid,
            "skipped": False,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features": int(len(keep)),
            "selected_features": keep,
            **metrics,
            "naive": naive,
        })
        preds_frames.append(pd.DataFrame({
            "Time": test_df.index,
            "patient_id": pid,
            "y_true": yte.to_numpy(),
            "y_pred": yhat,
        }))
    metrics_df = pd.DataFrame([r for r in fold_rows if not r.get("skipped")])
    summary = {}
    if not metrics_df.empty:
        for c in ("rmse", "mape", "accuracy"):
            summary[c] = {
                "mean": float(metrics_df[c].mean()),
                "std": float(metrics_df[c].std(ddof=1) if len(metrics_df) > 1 else 0.0),
            }
    return {
        "skipped": False,
        "n_participants": len(ids),
        "folds": fold_rows,
        "summary": summary,
        "predictions": pd.concat(preds_frames, ignore_index=True) if preds_frames else pd.DataFrame(),
        "importances": pd.concat(importances, ignore_index=True) if importances else pd.DataFrame(),
    }

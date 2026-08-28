"""Glucose-excursion classification (paper: Classification of glucose excursions).

Primary model: DecisionTreeClassifier + RFE (20 features) inside
RepeatedStratifiedKFold (10 splits, 3 repeats) on a class-balanced dataset.

Also: 70/30 train/test Decision Tree and Logistic Regression.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .constants import (
    DEFAULT_RANDOM_STATE,
    FEATURE_NAMES,
    KFOLD_REPEATS,
    KFOLD_SPLITS,
    LABEL_ORDER,
    RFE_N_FEATURES,
    TRAIN_TEST_FRACTION,
)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Paper: balanced accuracy, weighted precision/recall/F1, R² with negatives
    # clipped to 0 (Table 2 footnote).
    r2 = float(r2_score(y_true, y_pred))
    if r2 < 0:
        r2 = 0.0
    return {
        "balanced_accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "r2": r2,
    }


def balance_classes(
    df: pd.DataFrame,
    label_col: str = "label",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Undersample each class to the minority count (U14)."""
    labeled = df.dropna(subset=[label_col]).copy()
    counts_before = labeled[label_col].value_counts().to_dict()
    if labeled.empty:
        return labeled, {"n": 0, "counts_before": counts_before, "counts_after": {}}
    n = int(min(counts_before.values()))
    parts = []
    rng = np.random.RandomState(random_state)
    for lab in LABEL_ORDER:
        g = labeled[labeled[label_col] == lab]
        if g.empty:
            continue
        if len(g) > n:
            idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
            parts.append(g.loc[idx])
        else:
            parts.append(g)
    out = pd.concat(parts, axis=0).sort_index()
    info = {
        "method": "undersample_to_minority",
        "n_before": int(len(labeled)),
        "n_after": int(len(out)),
        "counts_before": {str(k): int(v) for k, v in counts_before.items()},
        "counts_after": {str(k): int(v) for k, v in out[label_col].value_counts().items()},
        "paper_N": 8666,
        "note": (
            "Paper N=8666 is the 16-patient balanced total. With a subset of "
            "patients the balanced N is 3 * minority_class_count."
        ),
    }
    return out, info


def _xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_NAMES].copy()
    # ID is numeric personalization; keep it. sklearn needs numeric dtypes.
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = df["label_int"].astype(int)
    return X, y


def _dt_pipeline(random_state: int) -> Pipeline:
    # U15: median impute for sklearn. U16: DecisionTree defaults. U19: RFE+DT.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rfe", RFE(
            estimator=DecisionTreeClassifier(random_state=random_state),
            n_features_to_select=RFE_N_FEATURES,
        )),
        ("clf", DecisionTreeClassifier(random_state=random_state)),
    ])


def _lr_pipeline(random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def run_repeated_kfold(
    balanced: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, Any]:
    X, y = _xy(balanced)
    n_splits = min(KFOLD_SPLITS, int(y.value_counts().min()))
    if n_splits < 2:
        return {
            "skipped": True,
            "reason": (
                f"Need >=2 samples per class for stratified k-fold; "
                f"class counts={y.value_counts().to_dict()}"
            ),
        }
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=KFOLD_REPEATS, random_state=random_state
    )
    fold_metrics: List[Dict[str, float]] = []
    selected: List[List[str]] = []
    for fold_i, (tr, te) in enumerate(cv.split(X, y)):
        pipe = _dt_pipeline(random_state + fold_i)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        m = _metrics(y.iloc[te].to_numpy(), pred)
        m["fold"] = fold_i
        fold_metrics.append(m)
        rfe: RFE = pipe.named_steps["rfe"]
        selected.append([c for c, keep in zip(FEATURE_NAMES, rfe.support_) if keep])
    dfm = pd.DataFrame(fold_metrics)
    summary = {c: {"mean": float(dfm[c].mean()), "std": float(dfm[c].std())}
               for c in ["balanced_accuracy", "recall", "precision", "f1", "r2"]}
    return {
        "skipped": False,
        "n_splits_used": n_splits,
        "n_repeats": KFOLD_REPEATS,
        "n_samples": int(len(balanced)),
        "summary": summary,
        "fold_metrics": fold_metrics,
        "rfe_features_last_fold": selected[-1] if selected else [],
    }


def run_holdout(
    balanced: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, Any]:
    X, y = _xy(balanced)
    counts = y.value_counts()
    if counts.min() < 2:
        return {"skipped": True, "reason": "Not enough samples per class for a stratified 70/30 split."}
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, train_size=TRAIN_TEST_FRACTION, stratify=y, random_state=random_state
    )
    out: Dict[str, Any] = {"skipped": False, "n_train": int(len(Xtr)), "n_test": int(len(Xte))}

    dt = _dt_pipeline(random_state)
    dt.fit(Xtr, ytr)
    pred_dt = dt.predict(Xte)
    out["decision_tree"] = _metrics(yte.to_numpy(), pred_dt)
    out["decision_tree"]["confusion"] = _confusion(yte.to_numpy(), pred_dt)
    rfe: RFE = dt.named_steps["rfe"]
    out["decision_tree"]["selected_features"] = [
        c for c, keep in zip(FEATURE_NAMES, rfe.support_) if keep
    ]

    lr = _lr_pipeline(random_state)
    lr.fit(Xtr, ytr)
    pred_lr = lr.predict(Xte)
    out["logistic_regression"] = _metrics(yte.to_numpy(), pred_lr)
    out["logistic_regression"]["confusion"] = _confusion(yte.to_numpy(), pred_lr)
    return out


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    labels = [0, 1, 2]
    names = LABEL_ORDER
    mat = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            mat[int(t), int(p)] += 1
    per_class = {}
    for i, name in enumerate(names):
        tot = mat[i].sum()
        per_class[name] = float(mat[i, i] / tot) if tot else None
    return {
        "order": names,
        "matrix": mat.tolist(),
        "per_class_accuracy": per_class,
    }


def run_classification(
    table: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, Any]:
    balanced, bal_info = balance_classes(table, random_state=random_state)
    return {
        "balancing": bal_info,
        "repeated_kfold": run_repeated_kfold(balanced, random_state=random_state),
        "holdout_70_30": run_holdout(balanced, random_state=random_state),
    }

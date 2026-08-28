"""Data loading for the PhysioNet/BIG IDEAs Lab per-patient folder layout.

Expected layout (paper cohort released as PhysioNet big-ideas-glycemic-wearable):

    Data/
      Demographics.csv          # optional; ID, Gender, HbA1c
      015/
        ACC_015.csv
        BVP_015.csv
        Dexcom_015.csv
        EDA_015.csv
        Food_Log_015.csv
        HR_015.csv
        IBI_015.csv
        TEMP_015.csv

Column names are taken from the files themselves, not assumed from Empatica
raw export conventions, except where the paper/SI/PhysioNet description
identifies them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import SENSOR_FILE_STEMS


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return project_root() / "Data"


def discover_patient_ids(data_dir: Path) -> List[str]:
    """Return sorted 3-digit patient folder names that contain Dexcom data."""
    ids = []
    if not data_dir.exists():
        return ids
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            dex = list(p.glob(f"Dexcom_{p.name}.*")) + list(p.glob("Dexcom.csv"))
            if dex:
                ids.append(p.name)
    return ids


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    return _strip_columns(df)


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


@dataclass
class SensorSeries:
    name: str
    path: Path
    n_rows: int
    time_min: Optional[pd.Timestamp]
    time_max: Optional[pd.Timestamp]
    sorted: bool
    value_cols: List[str]
    n_missing: Dict[str, int]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatientData:
    patient_id: str
    folder: Path
    glucose: pd.DataFrame  # index Time, column Glucose
    eda: pd.DataFrame      # index Time, column EDA
    temp: pd.DataFrame
    hr: pd.DataFrame
    acc: pd.DataFrame      # index Time, columns acc_x, acc_y, acc_z, ACC
    ibi: pd.DataFrame      # index Time, column IBI (seconds)
    food: pd.DataFrame
    bvp_inspect: Dict[str, Any]
    demographics: Dict[str, Any]
    file_reports: Dict[str, Any]
    food_alignment: Dict[str, Any]


def load_demographics(data_dir: Path, patient_id: str) -> Dict[str, Any]:
    """Load Biological Sex and HbA1c (SI Table 1). Source: Data/Demographics.csv.

    U02: this file is not inside the patient folder; it is the PhysioNet companion.
    U03: Female=0, Male=1.
    """
    path = data_dir / "Demographics.csv"
    out: Dict[str, Any] = {
        "path": str(path) if path.exists() else None,
        "Biological_Sex": np.nan,
        "HbA1c": np.nan,
        "Gender_raw": None,
        "found": False,
    }
    if not path.exists():
        return out
    demo = _read_csv(path)
    demo["ID"] = demo["ID"].astype(str).str.zfill(3)
    pid = str(patient_id).zfill(3)
    # PhysioNet uses unpadded IDs (15) as well.
    row = demo.loc[demo["ID"] == pid]
    if row.empty:
        row = demo.loc[demo["ID"].str.lstrip("0") == pid.lstrip("0")]
    if row.empty:
        return out
    r = row.iloc[0]
    out["found"] = True
    if "Gender" in r.index:
        g = str(r["Gender"]).strip().upper()
        out["Gender_raw"] = g
        if g in {"F", "FEMALE"}:
            out["Biological_Sex"] = 0.0
        elif g in {"M", "MALE"}:
            out["Biological_Sex"] = 1.0
    if "HbA1c" in r.index:
        out["HbA1c"] = float(pd.to_numeric(r["HbA1c"], errors="coerce"))
    return out


def _sensor_path(folder: Path, stem: str, patient_id: str) -> Optional[Path]:
    candidates = [
        folder / f"{stem}_{patient_id}.csv",
        folder / f"{stem}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def read_dexcom(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Dexcom G6 interstitial glucose (mg/dL), Event Type EGV only.

    Paper Methods, Dataset: records every 5 min.
    PhysioNet/file: Timestamp (YYYY-MM-DDThh:mm:ss), Glucose Value (mg/dL).
    """
    df = _read_csv(path)
    report: Dict[str, Any] = {
        "path": str(path),
        "n_rows_file": int(len(df)),
        "columns": list(df.columns),
        "event_type_counts": None,
    }
    if "Event Type" in df.columns:
        report["event_type_counts"] = df["Event Type"].value_counts(dropna=False).to_dict()
        df = df[df["Event Type"].astype(str).str.upper() == "EGV"].copy()
    tcol = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    if tcol not in df.columns:
        for c in ("Timestamp", "Time", "Datetime", "datetime"):
            if c in df.columns:
                tcol = c
                break
    gcol = "Glucose Value (mg/dL)"
    if gcol not in df.columns:
        for c in ("Glucose (mg/dL)", "Glucose", "glucose"):
            if c in df.columns:
                gcol = c
                break
    out = pd.DataFrame({
        "Time": _parse_datetime(df[tcol]),
        "Glucose": pd.to_numeric(df[gcol], errors="coerce"),
    }).dropna(subset=["Time"])
    out = out.sort_values("Time").drop_duplicates(subset=["Time"], keep="last")
    out = out.set_index("Time")
    report.update({
        "n_egv": int(len(out)),
        "n_glucose_missing": int(out["Glucose"].isna().sum()),
        "time_min": str(out.index.min()) if len(out) else None,
        "time_max": str(out.index.max()) if len(out) else None,
        "glucose_min": float(out["Glucose"].min()) if len(out) else None,
        "glucose_max": float(out["Glucose"].max()) if len(out) else None,
        "glucose_mean": float(out["Glucose"].mean()) if len(out) else None,
    })
    if len(out) > 1:
        diffs = pd.Series(out.index).diff().dt.total_seconds().dropna()
        report["interval_seconds_median"] = float(diffs.median())
        report["interval_seconds_mean"] = float(diffs.mean())
        report["n_gaps_gt_10min"] = int((diffs > 600).sum())
        report["n_gaps_gt_30min"] = int((diffs > 1800).sum())
    return out[["Glucose"]], report


def read_empatica_single(
    path: Path, value_name: str, expected_col: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = _read_csv(path)
    tcol = "datetime" if "datetime" in df.columns else df.columns[0]
    vcol = expected_col if expected_col in df.columns else df.columns[1]
    out = pd.DataFrame({
        "Time": _parse_datetime(df[tcol]),
        value_name: pd.to_numeric(df[vcol], errors="coerce"),
    }).dropna(subset=["Time"])
    was_sorted = bool(out["Time"].is_monotonic_increasing)
    out = out.sort_values("Time").set_index("Time")
    report = {
        "path": str(path),
        "n_rows": int(len(out)),
        "columns": list(df.columns),
        "value_col": vcol,
        "sorted_on_disk": was_sorted,
        "time_min": str(out.index.min()) if len(out) else None,
        "time_max": str(out.index.max()) if len(out) else None,
        "n_missing_value": int(out[value_name].isna().sum()),
        "value_min": float(out[value_name].min()) if len(out) else None,
        "value_max": float(out[value_name].max()) if len(out) else None,
        "value_mean": float(out[value_name].mean()) if len(out) else None,
        "dates_present": [str(d) for d in sorted(pd.Index(out.index.date).unique())],
    }
    if len(out) > 1:
        diffs = pd.Series(out.index).diff().dt.total_seconds().dropna()
        report["interval_seconds_median"] = float(diffs.median())
        report["n_gaps_gt_1h"] = int((diffs > 3600).sum())
    return out, report


def read_acc(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Tri-axial accelerometry; add vector magnitude ACC = sqrt(x^2+y^2+z^2).

    Paper Feature engineering: 'accelerometry (vector magnitude of the three axes)'.
    U22: no unit conversion.
    """
    df = _read_csv(
        path,
        dtype={"acc_x": np.float32, "acc_y": np.float32, "acc_z": np.float32},
    )
    tcol = "datetime" if "datetime" in df.columns else df.columns[0]
    for c in ("acc_x", "acc_y", "acc_z"):
        if c not in df.columns:
            raise ValueError(f"ACC file {path} missing {c}. Columns={list(df.columns)}")
    time = _parse_datetime(df[tcol])
    was_sorted = bool(time.is_monotonic_increasing)
    mag = np.sqrt(
        df["acc_x"].to_numpy(dtype=np.float32) ** 2
        + df["acc_y"].to_numpy(dtype=np.float32) ** 2
        + df["acc_z"].to_numpy(dtype=np.float32) ** 2
    )
    out = pd.DataFrame(
        {
            "Time": time,
            "acc_x": df["acc_x"].to_numpy(),
            "acc_y": df["acc_y"].to_numpy(),
            "acc_z": df["acc_z"].to_numpy(),
            "ACC": mag,
        }
    ).dropna(subset=["Time"])
    out = out.sort_values("Time").set_index("Time")
    report = {
        "path": str(path),
        "n_rows": int(len(out)),
        "sorted_on_disk": was_sorted,
        "time_min": str(out.index.min()) if len(out) else None,
        "time_max": str(out.index.max()) if len(out) else None,
        "acc_mag_min": float(out["ACC"].min()) if len(out) else None,
        "acc_mag_max": float(out["ACC"].max()) if len(out) else None,
        "acc_mag_mean": float(out["ACC"].mean()) if len(out) else None,
        "dates_present": [str(d) for d in sorted(pd.Index(out.index.date).unique())],
        "interval_seconds_median": 1.0 / 32.0,
    }
    return out, report


def inspect_bvp(path: Path) -> Dict[str, Any]:
    """BVP is recorded (PhysioNet) but is not a SI Table 1 feature (U20)."""
    report: Dict[str, Any] = {"path": str(path), "used_in_69_features": False}
    n = 0
    tmin = tmax = None
    dates = {}
    sorted_chunks = []
    for i, chunk in enumerate(pd.read_csv(path, usecols=[0, 1], chunksize=3_000_000)):
        chunk = _strip_columns(chunk)
        tcol = chunk.columns[0]
        t = _parse_datetime(chunk[tcol])
        n += int(t.notna().sum())
        if t.notna().any():
            cmin, cmax = t.min(), t.max()
            tmin = cmin if tmin is None else min(tmin, cmin)
            tmax = cmax if tmax is None else max(tmax, cmax)
            vc = t.dt.date.value_counts()
            for d, c in vc.items():
                dates[str(d)] = dates.get(str(d), 0) + int(c)
        sorted_chunks.append(bool(t.is_monotonic_increasing))
    report.update({
        "n_rows": n,
        "time_min": str(tmin) if tmin is not None else None,
        "time_max": str(tmax) if tmax is not None else None,
        "dates_present": sorted(dates.keys()),
        "n_per_date": dates,
        "sampling_hz_paper": 64,
        "sorted_on_disk_all_chunks": all(sorted_chunks) if sorted_chunks else None,
    })
    return report


def read_food(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = _read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "time_begin" in df.columns:
        t = _parse_datetime(df["time_begin"])
    else:
        t = _parse_datetime(df.iloc[:, 0])
    df = df.assign(Time=t).dropna(subset=["Time"]).sort_values("Time")
    for c in ("calorie", "total_carb", "sugar", "protein"):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    report = {
        "path": str(path),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "time_min": str(df["Time"].min()) if len(df) else None,
        "time_max": str(df["Time"].max()) if len(df) else None,
        "n_unique_dates": int(df["Time"].dt.date.nunique()) if len(df) else 0,
        "dates": [str(d) for d in sorted(df["Time"].dt.date.unique())] if len(df) else [],
        "n_calorie_zero": int((df["calorie"] == 0).sum()) if len(df) else 0,
        "n_time_end_missing": int(df["time_end"].isna().sum()) if "time_end" in df.columns else None,
    }
    return df, report


def align_food_to_glucose(
    food: pd.DataFrame, glucose: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Integer-day shift of food timestamps if they do not overlap CGM (U01)."""
    info: Dict[str, Any] = {
        "method": "as_recorded",
        "shift_days": 0,
        "overlap_before": False,
        "overlap_after": False,
        "note": "",
    }
    if food.empty or glucose.empty:
        info["note"] = "empty food or glucose; no alignment"
        return food, info
    g0, g1 = glucose.index.min(), glucose.index.max()
    f0, f1 = food["Time"].min(), food["Time"].max()
    overlap = (f1 >= g0) and (f0 <= g1)
    info["overlap_before"] = bool(overlap)
    info["food_range_before"] = [str(f0), str(f1)]
    info["glucose_range"] = [str(g0), str(g1)]
    if overlap:
        info["overlap_after"] = True
        info["note"] = "Food timestamps already overlap CGM; left unchanged."
        return food, info
    shift_days = int((g0.normalize() - f0.normalize()).days)
    food = food.copy()
    food["Time"] = food["Time"] + pd.Timedelta(days=shift_days)
    info["method"] = "shifted_to_cgm_start_date"
    info["shift_days"] = shift_days
    f0b, f1b = food["Time"].min(), food["Time"].max()
    info["food_range_after"] = [str(f0b), str(f1b)]
    info["overlap_after"] = bool((f1b >= g0) and (f0b <= g1))
    info["note"] = (
        "ASSUMED (U01): food log dates do not overlap CGM. Shifted by "
        f"{shift_days} days so the first food calendar day equals the first "
        "CGM calendar day. This is not described in the paper; it addresses "
        "a known PhysioNet date-shift inconsistency."
    )
    return food, info


def load_patient(data_dir: Path, patient_id: str, inspect_bvp: bool = True) -> PatientData:
    folder = data_dir / patient_id
    if not folder.is_dir():
        raise FileNotFoundError(f"Patient folder not found: {folder}")

    file_reports: Dict[str, Any] = {}
    missing_files = []

    dex_path = _sensor_path(folder, "Dexcom", patient_id)
    if dex_path is None:
        raise FileNotFoundError(f"No Dexcom file in {folder}")
    glucose, file_reports["Dexcom"] = read_dexcom(dex_path)

    def _req(stem: str, reader, *args):
        path = _sensor_path(folder, stem, patient_id)
        if path is None:
            missing_files.append(stem)
            return None, {"path": None, "missing": True}
        return reader(path, *args)

    eda, file_reports["EDA"] = _req("EDA", read_empatica_single, "EDA", "eda")
    temp, file_reports["TEMP"] = _req("TEMP", read_empatica_single, "TEMP", "temp")
    hr, file_reports["HR"] = _req("HR", read_empatica_single, "HR", "hr")
    acc, file_reports["ACC"] = _req("ACC", read_acc)
    ibi, file_reports["IBI"] = _req("IBI", read_empatica_single, "IBI", "ibi")

    food_path = folder / f"Food_Log_{patient_id}.csv"
    if not food_path.exists():
        food_path = folder / "Food_Log.csv"
    if food_path.exists():
        food, file_reports["Food_Log"] = read_food(food_path)
    else:
        food = pd.DataFrame()
        file_reports["Food_Log"] = {"missing": True}
        missing_files.append("Food_Log")

    food, food_alignment = align_food_to_glucose(food, glucose)

    bvp_path = _sensor_path(folder, "BVP", patient_id)
    if bvp_path is not None and inspect_bvp:
        file_reports["BVP"] = inspect_bvp(bvp_path)
        bvp_inspect = file_reports["BVP"]
    else:
        bvp_inspect = {"missing": bvp_path is None, "used_in_69_features": False}
        file_reports["BVP"] = bvp_inspect

    empty = pd.DataFrame()
    demographics = load_demographics(data_dir, patient_id)
    file_reports["missing_sensor_files"] = missing_files
    file_reports["n_files_present"] = int(sum(
        1 for s in SENSOR_FILE_STEMS + ["Food_Log"]
        if (folder / f"{s}_{patient_id}.csv").exists() or (folder / f"{s}.csv").exists()
        or (s == "Food_Log" and (folder / f"Food_Log_{patient_id}.csv").exists())
    ))

    return PatientData(
        patient_id=str(patient_id).zfill(3) if str(patient_id).isdigit() else str(patient_id),
        folder=folder,
        glucose=glucose if glucose is not None else empty,
        eda=eda if eda is not None else empty,
        temp=temp if temp is not None else empty,
        hr=hr if hr is not None else empty,
        acc=acc if acc is not None else empty,
        ibi=ibi if ibi is not None else empty,
        food=food,
        bvp_inspect=bvp_inspect,
        demographics=demographics,
        file_reports=file_reports,
        food_alignment=food_alignment,
    )


def json_default(obj: Any):
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")

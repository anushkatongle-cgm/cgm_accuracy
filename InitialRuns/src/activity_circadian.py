"""Activity-bout and circadian features (paper Methods; SI Table 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def activity_features(features: pd.DataFrame) -> pd.DataFrame:
    """Activity_bouts, Activity1hr, Activity24, ACC_mean_2hrs, ACC_max_2hrs.

    Paper: an interval is an activity bout if both 5-min mean ACC and mean HR
    exceed the average of prior historical data. Then rolling total bouts in
    the last hour and average bouts in the previous 24 h. Also mean and max
    accelerometry vector magnitude over the previous two hours.

    U10: expanding mean of all prior 5-min epochs, shifted by 1.
    """
    out = features.copy()
    if {"ACC_Mean", "HR_Mean"}.issubset(out.columns):
        prior_acc = out["ACC_Mean"].expanding(min_periods=1).mean().shift(1)
        prior_hr = out["HR_Mean"].expanding(min_periods=1).mean().shift(1)
        bouts = ((out["ACC_Mean"] > prior_acc) & (out["HR_Mean"] > prior_hr)).astype(float)
        bouts[prior_acc.isna() | prior_hr.isna()] = np.nan
        out["Activity_bouts"] = bouts
        out["Activity1hr"] = bouts.rolling("1h", min_periods=1).sum()
        out["Activity24"] = bouts.rolling("24h", min_periods=1).mean()
    if "ACC_Mean" in out.columns:
        # SI: mean of 5-min means over 2 h; max of 5-min maxima over 2 h.
        out["ACC_mean_2hrs"] = out["ACC_Mean"].rolling("2h", min_periods=1).mean()
    if "ACC_Max" in out.columns:
        out["ACC_max_2hrs"] = out["ACC_Max"].rolling("2h", min_periods=1).max()
    return out


def circadian_clock_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Minfrommid and Hourfrommid (paper: minutes/hours from midnight).

    U11/paper: minutes from midnight = hour*60 + minute.
    Hourfrommid: fractional hours (hour + minute/60). Integer vs fractional
    is not stated; fractional preserves within-hour timing.
    """
    return pd.DataFrame(
        {
            "Minfrommid": index.hour * 60 + index.minute,
            "Hourfrommid": index.hour + index.minute / 60.0,
        },
        index=index,
    )


def wake_time_feature(features: pd.DataFrame) -> pd.Series:
    """WakeTime in minutes after midnight (SI Table 1; paper Methods).

    Paper:
      For each day, mark a 5-min interval 0 if two of four (ACC_Mean, ACC_Std,
      HR_Mean, HR_Std) are below that day's average; else 1. Average over 3 h.
      Wake time is when the slope of this series sharply changes and remains
      consistently higher 25 and 75 min later.

    U11: paper polarity (>=2 of 4 below daily mean → 0). Search window
    04:00–14:00 is ASSUMED (not in the paper).
    """
    need = {"ACC_Mean", "ACC_Std", "HR_Mean", "HR_Std"}
    if not need.issubset(features.columns):
        return pd.Series(np.nan, index=features.index, name="WakeTime")

    df = features[list(need)].copy()
    df["_Date"] = df.index.normalize()
    day_mean = df.groupby("_Date")[list(need)].transform("mean")
    n_below = (
        (df["ACC_Mean"] < day_mean["ACC_Mean"]).astype(int)
        + (df["ACC_Std"] < day_mean["ACC_Std"]).astype(int)
        + (df["HR_Mean"] < day_mean["HR_Mean"]).astype(int)
        + (df["HR_Std"] < day_mean["HR_Std"]).astype(int)
    )
    # Paper: two of the four below average → 0, else 1.
    wake_sleep = np.where(df["HR_Mean"].isna(), np.nan, np.where(n_below >= 2, 0.0, 1.0))
    ws = pd.Series(wake_sleep, index=df.index, dtype=float)
    roll3 = ws.rolling("3h", min_periods=1).mean()

    step = pd.Timedelta("5min")
    n25 = int(pd.Timedelta("25min") / step)
    n75 = int(pd.Timedelta("75min") / step)

    wake_by_day = {}
    for day, g in roll3.groupby(roll3.index.normalize()):
        w = g.between_time("04:00", "14:00")
        if w.dropna().shape[0] < 12:
            wake_by_day[day] = np.nan
            continue
        vals = w.to_numpy(dtype=float)
        idx = w.index
        found = np.nan
        for i in range(0, len(vals) - n75):
            if not np.isfinite(vals[i]):
                continue
            later25 = vals[i + n25] if i + n25 < len(vals) else np.nan
            later75 = vals[i + n75]
            prev = vals[i - 1] if i > 0 else vals[i]
            slope_up = np.isfinite(prev) and (vals[i] - prev) >= 0
            if (
                np.isfinite(later25)
                and np.isfinite(later75)
                and later25 > vals[i]
                and later75 > vals[i]
                and slope_up
            ):
                found = float(idx[i].hour * 60 + idx[i].minute)
                break
        wake_by_day[day] = found

    mapped = df["_Date"].map(wake_by_day)
    return pd.Series(mapped.to_numpy(dtype=float), index=features.index, name="WakeTime")

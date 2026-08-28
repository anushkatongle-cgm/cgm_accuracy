"""Stress features: EDA peaks and HRV (paper Methods; SI Table 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .constants import (
    EDA_FS_HZ,
    EDA_PEAK_DISTANCE_SAMPLES,
    EDA_PEAK_HEIGHT,
    EDA_PEAK_PROMINENCE_US,
    NN50_MS,
)
from .sensor_stats import contiguous_sessions


def eda_peak_features(eda: pd.DataFrame) -> pd.DataFrame:
    """PeakEDA, PeakEDA2hr_sum, PeakEDA2hr_mean.

    Paper: SciPy find_peaks, distance=1 s (4 samples), prominence=0.3 µS.
    SI Table 1 also lists height=0. Count peaks per 5-min interval; 2 h rolling
    sum and mean of those counts.

    U08: peaks are detected within contiguous sessions so Empatica recording
    gaps are not treated as adjacent samples.
    """
    if eda is None or eda.empty:
        return pd.DataFrame(columns=["PeakEDA", "PeakEDA2hr_sum", "PeakEDA2hr_mean"])
    s = eda["EDA"].sort_index()
    idx = s.index
    sessions = contiguous_sessions(idx, max_gap=pd.Timedelta(seconds=2.0 / EDA_FS_HZ))
    values = s.to_numpy(dtype=np.float64)
    peak_times = []
    for sid in np.unique(sessions):
        mask = sessions == sid
        x = values[mask]
        t = idx[mask]
        if x.size < EDA_PEAK_DISTANCE_SAMPLES + 1:
            continue
        peaks, _ = find_peaks(
            x,
            height=EDA_PEAK_HEIGHT,
            distance=EDA_PEAK_DISTANCE_SAMPLES,
            prominence=EDA_PEAK_PROMINENCE_US,
        )
        if peaks.size:
            peak_times.append(t[peaks])
    if peak_times:
        peak_index = pd.DatetimeIndex(np.concatenate([np.asarray(t) for t in peak_times]))
        peak_series = pd.Series(1, index=peak_index, name="Peak").sort_index()
        peak_per_bin = peak_series.resample("5min").sum()
    else:
        start = idx.min().floor("5min")
        end = idx.max().floor("5min")
        peak_per_bin = pd.Series(0.0, index=pd.date_range(start, end, freq="5min"), name="Peak")

    # Fill bins that exist in the EDA recording but had zero peaks.
    eda_bins = s.resample("5min").size()
    peak_per_bin = peak_per_bin.reindex(eda_bins.index).fillna(0.0)
    df = pd.DataFrame({"PeakEDA": peak_per_bin.astype(float)})
    df["PeakEDA2hr_sum"] = df["PeakEDA"].rolling("2h", min_periods=1).sum()
    df["PeakEDA2hr_mean"] = df["PeakEDA"].rolling("2h", min_periods=1).mean()
    return df


def hrv_features(ibi: pd.DataFrame) -> pd.DataFrame:
    """8 HRV metrics per 5-min interval from IBI (seconds in the file).

    Paper: mean, median, max, min HRV, SDNN, RMSSD, NN50, pNN50.
    U09: convert IBI to milliseconds; RMSSD uses rms of successive diffs;
    pNN50 = NN50 / n_IBI (SI); SDNN = sample std (ddof=1).
    """
    cols = ["maxHRV", "minHRV", "medianHRV", "meanHRV", "SDNN", "NN50", "pNN50", "RMSSD"]
    if ibi is None or ibi.empty:
        return pd.DataFrame(columns=cols)
    s = ibi["IBI"].sort_index()
    s = s[np.isfinite(s.to_numpy()) & (s.to_numpy() > 0)]
    rows = []
    for t, g in s.groupby(pd.Grouper(freq="5min")):
        if g.empty:
            continue
        ibi_ms = g.to_numpy(dtype=np.float64) * 1000.0
        n = ibi_ms.size
        diffs = np.diff(ibi_ms)
        abs_diffs = np.abs(diffs)
        nn50 = float(np.sum(abs_diffs > NN50_MS)) if diffs.size else np.nan
        pnn50 = float(nn50 / n) if n else np.nan  # SI: NN50 / len(N)
        if diffs.size:
            rmssd = float(np.sqrt(np.mean(diffs ** 2)))
        else:
            rmssd = np.nan
        sdnn = float(np.std(ibi_ms, ddof=1)) if n > 1 else np.nan
        rows.append({
            "Time": t,
            "maxHRV": float(np.max(ibi_ms)),
            "minHRV": float(np.min(ibi_ms)),
            "medianHRV": float(np.median(ibi_ms)),
            "meanHRV": float(np.mean(ibi_ms)),
            "SDNN": sdnn,
            "NN50": nn50,
            "pNN50": pnn50,
            "RMSSD": rmssd,
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).set_index("Time").sort_index()

"""5-minute data-driven summary statistics (SI Table 1).

Paper Methods, Feature engineering:
  'Data-driven features for each of the 5-min intervals of smart watch data
   include 7 summary statistics for each sensor: mean, standard deviation,
   minimum, maximum, first quartile, third quartile, and skew.'

SI Table 1 formulas (followed exactly):
  mean μ = sum(x_i) / N
  std  σ = sqrt(sum((x_i - μ)^2) / N)     # population, divide by N
  skew   = sum((x_i - xbar)^3) / ((N-1) * σ^3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _si_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return np.nan
    mu = x.mean()
    # SI σ uses N in the denominator.
    var_pop = np.mean((x - mu) ** 2)
    if var_pop <= 0:
        return 0.0
    sigma = np.sqrt(var_pop)
    return float(np.sum((x - mu) ** 3) / ((n - 1) * sigma ** 3))


def _si_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 1:
        return np.nan
    mu = x.mean()
    return float(np.sqrt(np.mean((x - mu) ** 2)))


def _q(arr: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(arr, q, method="linear"))
    except TypeError:
        return float(np.quantile(arr, q, interpolation="linear"))


def five_min_stats(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Compute SI 7-summary stats on 5-min bins of a datetime-indexed series."""
    if series is None or series.empty:
        return pd.DataFrame(columns=[
            f"{prefix}_Mean", f"{prefix}_Std", f"{prefix}_Min", f"{prefix}_Max",
            f"{prefix}_Q1G", f"{prefix}_Q3G", f"{prefix}_Skew",
        ])
    s = series.sort_index()
    s = s[np.isfinite(s.to_numpy())]
    if s.empty:
        return pd.DataFrame()

    return _fast_bin_stats(s, prefix)


def _fast_bin_stats(s: pd.Series, prefix: str) -> pd.DataFrame:
    idx = s.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    bins = idx.floor("5min")
    values = s.to_numpy(dtype=np.float64)
    df = pd.DataFrame({"x": values}, index=bins)
    df = df[np.isfinite(df["x"].to_numpy())]
    g = df.groupby(level=0, sort=True)["x"]
    out = pd.DataFrame({
        f"{prefix}_Mean": g.mean(),
        f"{prefix}_Std": g.std(ddof=0),
        f"{prefix}_Min": g.min(),
        f"{prefix}_Max": g.max(),
        f"{prefix}_Q1G": g.quantile(0.25),
        f"{prefix}_Q3G": g.quantile(0.75),
        f"{prefix}_Skew": g.apply(lambda x: _si_skew(x.to_numpy())),
        f"{prefix}_N": g.size(),
    })
    out.index.name = "Time"
    return out.sort_index()


def contiguous_sessions(index: pd.DatetimeIndex, max_gap: pd.Timedelta) -> np.ndarray:
    """Return integer session ids, splitting where the time gap exceeds max_gap."""
    if len(index) == 0:
        return np.array([], dtype=int)
    deltas = pd.Series(index).diff()
    new_session = (deltas > max_gap).fillna(False).to_numpy()
    return np.cumsum(new_session)

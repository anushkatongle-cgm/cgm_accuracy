"""Personalized glucose excursion labels (paper Methods; Fig. 3; Table 1).

PersHigh: glucose > mean(last 24 h) + 1 SD
PersLow:  glucose < mean(last 24 h) - 1 SD
PersNorm: within ±1 SD of the 24 h mean

U13: a full 24 h of elapsed lookback is required before a label is assigned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import LABEL_TO_INT


def personalized_labels(glucose: pd.Series) -> pd.DataFrame:
    """Return DataFrame with glucose_mean_24h, glucose_std_24h, label, label_int."""
    g = glucose.sort_index().astype(float)
    # Rolling 24 h on a DatetimeIndex uses time-based windows.
    mean24 = g.rolling("24h", min_periods=1).mean()
    std24 = g.rolling("24h", min_periods=2).std(ddof=1)
    elapsed = g.index - g.index[0] if len(g) else pd.TimedeltaIndex([])
    enough = pd.Series(elapsed >= pd.Timedelta("24h"), index=g.index) if len(g) else pd.Series(dtype=bool)

    high = g > (mean24 + std24)
    low = g < (mean24 - std24)
    label = pd.Series(np.nan, index=g.index, dtype=object)
    label.loc[enough & high] = "PersHigh"
    label.loc[enough & low] = "PersLow"
    label.loc[enough & ~high & ~low] = "PersNorm"

    out = pd.DataFrame({
        "Glucose": g,
        "glucose_mean_24h": mean24,
        "glucose_std_24h": std24,
        "label": label,
        "label_int": label.map(LABEL_TO_INT),
        "label_ready": enough,
    })
    return out

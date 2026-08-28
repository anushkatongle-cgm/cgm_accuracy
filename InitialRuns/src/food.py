"""Diet features from the food log (paper Methods; SI Table 1)."""

from __future__ import annotations

import pandas as pd


def food_features(food: pd.DataFrame, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Rolling 2/8/24 h nutrient sums, Eat flag, Eat counts and means.

    Paper: rolling sum of calories, protein, carbohydrates, sugar over 2, 8, 24 h.
    Binary Eat=1 on intervals with a unique meal/snack/caloric beverage; rolling
    sum and mean of Eat over 2, 8, 24 h.

    U12: Eat=1 on the 5-min bin of time_begin for every log row; nutrients summed
    per timestamp; time_end duration is not filled.
    """
    cols = []
    for hours in (2, 8, 24):
        for c in ("calories", "protein", "sugar", "carbs"):
            cols.append(f"{c}{hours}hr")
    cols += [
        "Eat", "Eatcnt2hr", "Eatcnt8hr", "Eatcnt24hr",
        "Eatmean2hr", "Eatmean8hr", "Eatmean24hr",
    ]
    base = pd.DataFrame(index=base_index)
    if food is None or food.empty or base_index is None or len(base_index) == 0:
        return pd.DataFrame(0.0, index=base_index, columns=cols) if len(base_index) else pd.DataFrame(columns=cols)

    df = food.copy()
    agg = (
        df.groupby("Time", as_index=True)[["calorie", "total_carb", "sugar", "protein"]]
        .sum()
        .sort_index()
        .rename(columns={"calorie": "calories", "total_carb": "carbs"})
    )
    # Instantaneous 5-min sums on the modeling grid.
    inst = agg.resample("5min").sum()
    x = base.join(inst, how="left").fillna(0.0)

    for hours in (2, 8, 24):
        w = f"{hours}h"
        x[f"calories{hours}hr"] = x["calories"].rolling(w, min_periods=1).sum()
        x[f"protein{hours}hr"] = x["protein"].rolling(w, min_periods=1).sum()
        x[f"sugar{hours}hr"] = x["sugar"].rolling(w, min_periods=1).sum()
        x[f"carbs{hours}hr"] = x["carbs"].rolling(w, min_periods=1).sum()

    # Eat: any logged consumption in the 5-min bin (not calorie>0 only; U12).
    eat_times = df["Time"].drop_duplicates().sort_values()
    eat_bins = pd.Series(1, index=eat_times.dt.floor("5min")).groupby(level=0).max()
    x["Eat"] = eat_bins.reindex(x.index).fillna(0).astype(int)
    for hours in (2, 8, 24):
        w = f"{hours}h"
        x[f"Eatcnt{hours}hr"] = x["Eat"].rolling(w, min_periods=1).sum()
        x[f"Eatmean{hours}hr"] = x["Eat"].rolling(w, min_periods=1).mean()

    return x[cols]

"""Helpers for CASEN exploration (BFM income replication).

Logic lives here; plotting lives in figures.py.
"""
import pandas as pd
import numpy as np
from pathlib import Path


# Canonical columns for BFM income exercise
INCOME_COLS = ["yaut", "ytot", "yautcor", "ytotcor", "yauth", "ytoth", "yautcorh", "ytotcorh", "ypchtotcor"]
WEIGHT_COL = "expr"
HH_COLS = ["folio", "hogar"]


def load_casen(path, convert_categoricals=False):
    """Load CASEN stata with safe defaults for duplicate labels.

    Args:
        path (str|Path): Path to .dta.
        convert_categoricals (bool): Keep False to avoid rama4_sub_rev3 error.

    Returns:
        pd.DataFrame
    """
    return pd.read_stata(path, convert_categoricals=convert_categoricals)


def income_overview(df):
    """Which income columns exist and their missingness.

    Args:
        df (pd.DataFrame): CASEN dataframe.

    Returns:
        pd.DataFrame: table with n, n_missing, pct_missing, mean, median
    """
    rows = []
    for c in INCOME_COLS:
        if c not in df.columns:
            continue
        s = df[c]
        rows.append({
            "column": c,
            "n": int(s.notna().sum()),
            "n_missing": int(s.isna().sum()),
            "pct_missing": float(s.isna().mean() * 100),
            "mean": float(s.mean(skipna=True)),
            "median": float(s.median(skipna=True)),
            "min": float(s.min(skipna=True)),
            "max": float(s.max(skipna=True)),
        })
    return pd.DataFrame(rows).set_index("column")


def summarize_income(df, income_col="ytotcor", weight_col="expr"):
    """Weighted summary for BFM variable.

    Args:
        df (pd.DataFrame): CASEN.
        income_col (str): Column to summarize (household income).
        weight_col (str): Expansion factor.

    Returns:
        dict: stats
    """
    sub = df[[income_col, weight_col]].dropna(subset=[income_col])
    # at person level each row is a person; for household income use household-level vars like ytotcorh
    # weighted stats
    x = sub[income_col].values
    w = sub[weight_col].values.astype(float)
    # weighted quantiles
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cum = np.cumsum(ws) / ws.sum()
    def wq(p): return xs[np.searchsorted(cum, p)]
    return {
        "n_obs": int(len(sub)),
        "n_missing": int(df[income_col].isna().sum()),
        "weighted_mean": float(np.average(x, weights=w)),
        "weighted_median": float(wq(0.5)),
        "p10": float(wq(0.10)),
        "p25": float(wq(0.25)),
        "p75": float(wq(0.75)),
        "p90": float(wq(0.90)),
        "p99": float(wq(0.99)),
        "share_zero": float((x == 0).mean() * 100),
        "max": float(x.max()),
    }


def to_household(df, income_col="ytotcorh", weight_col="expr"):
    """Collapse to household level (one row per folio/hogar).

    CASEN is person-level; household income vars are repeated per person.
    We keep first row per household.

    Args:
        df (pd.DataFrame): person-level CASEN.
        income_col (str): household income col (h suffix).
        weight_col (str): weight.

    Returns:
        pd.DataFrame: household-level
    """
    cols = [c for c in ["folio", "hogar", income_col, weight_col, "tot_per_h", "region", "area"] if c in df.columns]
    hh = df[cols].drop_duplicates(subset=["folio", "hogar"])
    return hh

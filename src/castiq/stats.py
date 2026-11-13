"""Small statistics helpers. Every rate in the report carries a confidence interval and a sample size."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Stable for small n and for rates near 0 or 1."""
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rate_table(df: pd.DataFrame, by: str, outcome: str, min_n: int = 0) -> pd.DataFrame:
    """Win (or conversion) rate per group, with n, 95% CI, and lift in points vs the overall rate."""
    overall = float(df[outcome].mean()) if len(df) else float("nan")
    g = df.groupby(by, dropna=False, observed=True)[outcome].agg(["sum", "count"]).reset_index()
    g.columns = [by, "successes", "n"]
    g = g[g["n"] >= min_n].copy()
    ci = [wilson_interval(int(s), int(n)) for s, n in zip(g["successes"], g["n"], strict=True)]
    g["rate"] = g["successes"] / g["n"]
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]
    g["lift_pts"] = (g["rate"] - overall) * 100
    # A group "differs" when its CI does not contain the overall rate.
    g["significant"] = (g["ci_low"] > overall) | (g["ci_high"] < overall)
    return g.sort_values("rate", ascending=False, kind="mergesort").reset_index(drop=True)


def coefficient_of_variation(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or arr.mean() == 0:
        return float("nan")
    return float(arr.std(ddof=0) / arr.mean())


def gini(values: np.ndarray | pd.Series) -> float:
    """Gini coefficient: 0 = perfectly even, 1 = all in one bucket."""
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0 or arr.sum() == 0:
        return float("nan")
    n = arr.size
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)

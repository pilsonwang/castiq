"""ICP fit scoring and market sizing (TAM / SAM / SOM) on real US firmographics.

The score is a transparent weighted sum, not a black box: a sales leader can read config/icp.yaml and
predict the score of any company by hand. That is a deliberate choice - reps ignore scores they can't explain.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TIER_ORDER = ["A", "B", "C", "D"]


def assign_tier(score: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    conditions = [score >= thresholds["A"], score >= thresholds["B"], score >= thresholds["C"]]
    tiers = np.select(conditions, ["A", "B", "C"], default="D")
    return pd.Series(pd.Categorical(tiers, categories=TIER_ORDER, ordered=True), index=score.index)


def score_companies(companies: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add industry_fit, size_fit, geo_fit, fit_score (0-100), tier, and est_annual_spend_usd.

    Companies whose industry is not in the ICP get industry_fit 0 and are dropped: they are not in the market.
    """
    w = cfg["weights"]
    industry_fit = {k.lower(): float(v) for k, v in cfg["industry_fit"].items()}
    size_fit = {str(k): float(v) for k, v in cfg["size_fit"].items()}
    geo_states = {k: float(v) for k, v in cfg["geo_fit"]["states"].items()}
    geo_default = float(cfg["geo_fit"]["default"])
    spend = {str(k): float(v) for k, v in cfg["annual_spend_usd"].items()}

    df = companies.copy()
    df["size_band"] = df["size_band"].fillna("unknown").astype(str)
    df["industry_fit"] = df["industry"].str.lower().map(industry_fit).fillna(0.0)
    df = df[df["industry_fit"] > 0].copy()
    df["size_fit"] = df["size_band"].map(size_fit).fillna(size_fit.get("unknown", 0.0))
    # Unknown state: we can't plan a visit, so use a value below the default.
    df["geo_fit"] = df["state"].map(geo_states).fillna(geo_default)
    df.loc[df["state"].isna(), "geo_fit"] = geo_default * 0.5
    df["fit_score"] = (
        100 * (w["industry"] * df["industry_fit"] + w["size"] * df["size_fit"] + w["geo"] * df["geo_fit"])
    ).round(1)
    df["tier"] = assign_tier(df["fit_score"], cfg["tiers"])
    df["est_annual_spend_usd"] = df["size_band"].map(spend).fillna(0.0) * df["industry_fit"]
    # Deterministic order: ties on score are broken by spend, then name and handle, so the target-account
    # list is identical on every run (DuckDB does not guarantee row order after a join).
    return df.sort_values(["fit_score", "est_annual_spend_usd", "name", "handle"],
                          ascending=[False, False, True, True], kind="mergesort").reset_index(drop=True)


def market_sizing(scored: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """TAM = all in-market spend. SAM = A-C tier spend x serviceable share. SOM = SAM x obtainable share."""
    tam = float(scored["est_annual_spend_usd"].sum())
    sam_base = float(scored.loc[scored["tier"].isin(["A", "B", "C"]), "est_annual_spend_usd"].sum())
    sam = sam_base * float(cfg["serviceable_share"])
    som = sam * float(cfg["obtainable_share"])
    by_tier = (
        scored.groupby("tier", observed=False)
        .agg(accounts=("handle", "count"), spend_usd=("est_annual_spend_usd", "sum"),
             avg_fit=("fit_score", "mean"))
        .reindex(TIER_ORDER)
        .fillna(0)
        .reset_index()
    )
    return {
        "in_market_accounts": len(scored),
        "tam_usd": tam,
        "sam_usd": sam,
        "som_usd": som,
        "by_tier": by_tier,
        "tier_a_accounts": int((scored["tier"] == "A").sum()),
        "tier_ab_accounts": int(scored["tier"].isin(["A", "B"]).sum()),
    }


def industry_summary(scored: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    g = (
        scored.groupby("industry")
        .agg(accounts=("handle", "count"),
             tier_ab=("tier", lambda s: int(s.isin(["A", "B"]).sum())),
             spend_usd=("est_annual_spend_usd", "sum"),
             industry_fit=("industry_fit", "first"))
        .sort_values("spend_usd", ascending=False, kind="mergesort")
        .head(top)
        .reset_index()
    )
    return g


def state_summary(scored: pd.DataFrame, tiers: list[str]) -> pd.DataFrame:
    sub = scored[scored["tier"].isin(tiers) & scored["state"].notna()]
    return (
        sub.groupby("state")
        .agg(accounts=("handle", "count"), potential_usd=("est_annual_spend_usd", "sum"))
        .sort_values("potential_usd", ascending=False, kind="mergesort")
        .reset_index()
    )

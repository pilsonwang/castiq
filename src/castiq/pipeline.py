"""Pipeline health: funnel, win rates by segment (with confidence intervals), stalled deals, rep spread."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from castiq.stats import rate_table, wilson_interval

SEGMENTS = {
    "lead_source": "Lead source",
    "industry": "Industry",
    "employee_band": "Company size",
    "region": "Region",
    "buyer_role": "Buyer role of first contact",
    "seniority": "Seniority of first contact",
    "buyer_roles_engaged": "Buyer roles engaged at account",
    "process_maturity_band": "Process maturity",
}


def summary(opps: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    closed = opps[opps["is_closed"]]
    open_ = opps[~opps["is_closed"]]
    won = closed[closed["is_won"]]
    lost = closed[~closed["is_won"]]
    lo, hi = wilson_interval(len(won), len(closed))
    stalled = open_[open_["days_since_engaged"] >= cfg["stalled_after_days"]]
    return {
        "as_of_date": str(pd.to_datetime(opps["as_of_date"]).max().date()),
        "opportunities": len(opps),
        "open_opportunities": len(open_),
        "closed_opportunities": len(closed),
        "won": len(won),
        "win_rate": len(won) / len(closed) if len(closed) else float("nan"),
        "win_rate_ci": [lo, hi],
        "open_pipeline_acv": float(open_["acv"].sum()),
        "won_acv": float(won["acv"].sum()),
        "avg_won_acv": float(won["acv"].mean()),
        "avg_lost_acv": float(lost["acv"].mean()),
        "median_open_acv": float(open_["acv"].median()),
        "stalled_deals": len(stalled),
        "stalled_acv": float(stalled["acv"].sum()),
        "stalled_share_of_pipeline": float(stalled["acv"].sum() / open_["acv"].sum()) if len(open_) else 0.0,
        "stalled_after_days": int(cfg["stalled_after_days"]),
    }


def segment_win_rates(opps: pd.DataFrame, min_n: int) -> dict[str, pd.DataFrame]:
    closed = opps[opps["is_closed"]].copy()
    closed["won"] = closed["is_won"].astype(int)
    return {label: rate_table(closed, col, "won", min_n=min_n) for col, label in SEGMENTS.items()
            if col in closed.columns}


def stage_mix(opps: pd.DataFrame) -> pd.DataFrame:
    order = ["demo_scheduled", "demo_completed", "proposal_sent", "negotiation", "closed_won", "closed_lost"]
    g = opps.groupby("stage").agg(deals=("opportunity_id", "count"), acv=("acv", "sum")).reindex(order)
    return g.dropna(how="all").fillna(0).reset_index()


def rep_spread(opps: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Win rate per rep, and a chi-square test: are reps really different, or is it noise?"""
    closed = opps[opps["is_closed"]].copy()
    closed["won"] = closed["is_won"].astype(int)
    reps = rate_table(closed, "owner_rep_id", "won")
    open_ = opps[~opps["is_closed"]]
    stalled = open_[open_["days_since_engaged"] >= cfg["stalled_after_days"]]
    extra = pd.DataFrame({
        "owner_rep_id": open_.groupby("owner_rep_id").size().index,
        "open_deals": open_.groupby("owner_rep_id").size().to_numpy(),
    }).merge(stalled.groupby("owner_rep_id").size().rename("stalled_deals").reset_index(), how="left")
    extra = extra.merge(open_.groupby("owner_rep_id")["acv"].sum().rename("open_acv").reset_index())
    reps = reps.merge(extra, on="owner_rep_id", how="left").fillna({"stalled_deals": 0})
    table = pd.crosstab(closed["owner_rep_id"], closed["won"])
    chi2, p, dof, _ = chi2_contingency(table)
    return reps, {"chi2": float(chi2), "p_value": float(p), "dof": int(dof),
                  "min_rate": float(reps["rate"].min()), "max_rate": float(reps["rate"].max())}


def multithreading_effect(opps: pd.DataFrame) -> pd.DataFrame:
    closed = opps[opps["is_closed"]].copy()
    closed["won"] = closed["is_won"].astype(int)
    closed["roles"] = np.where(closed["buyer_roles_engaged"] >= 3, "3+",
                               closed["buyer_roles_engaged"].astype(int).astype(str))
    return rate_table(closed, "roles", "won").sort_values("roles").reset_index(drop=True)

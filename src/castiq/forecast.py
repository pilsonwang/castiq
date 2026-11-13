"""Forecast: calibrated deal-level win probability + Monte Carlo range, compared with CRM stage defaults.

Close dates are hidden in the source data, so a back-test over past quarters is not possible. Instead the
model is validated where we *can* check it: out-of-fold on every closed deal (AUC, Brier, calibration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["acv", "touches", "inbound_touches", "outbound_touches", "sessions", "pricing_page_views",
           "demo_page_views", "activities", "meetings", "demos_completed", "live_conversations",
           "buyer_roles_engaged", "contacts_engaged", "days_lead_to_opp"]
CATEGORICAL = ["industry", "region", "employee_band", "estimated_revenue_band", "process_maturity_band",
               "lead_source", "seniority", "buyer_role", "role_function"]


@dataclass
class ForecastResult:
    metrics: dict[str, float]
    calibration: pd.DataFrame
    open_deals: pd.DataFrame
    simulation: np.ndarray
    by_rep: pd.DataFrame


def _model() -> Pipeline:
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=0.3, max_iter=5000))])


def _xy(df: pd.DataFrame) -> pd.DataFrame:
    x = df[NUMERIC + CATEGORICAL].copy()
    for c in CATEGORICAL:
        x[c] = x[c].astype("object").where(x[c].notna(), "missing").astype(str)
    for c in NUMERIC:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)
    return x


def run(opps: pd.DataFrame, cfg: dict[str, Any]) -> ForecastResult:
    seed = int(cfg["random_seed"])
    closed = opps[opps["is_closed"]].reset_index(drop=True)
    open_ = opps[~opps["is_closed"]].reset_index(drop=True)
    y = closed["is_won"].astype(int).to_numpy()
    x = _xy(closed)

    # 1. Out-of-fold validation on closed deals.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(_model(), x, y, cv=cv, method="predict_proba")[:, 1]
    base = float(y.mean())
    frac_pos, mean_pred = calibration_curve(y, oof, n_bins=8, strategy="quantile")
    calib = pd.DataFrame({"predicted": mean_pred, "actual": frac_pos})
    stage_p = {k: float(v) for k, v in cfg["default_stage_probability"].items()}

    # 2. Fit on all closed deals, score the open pipeline.
    model = _model().fit(x, y)
    scored = open_.copy()
    scored["p_win_model"] = model.predict_proba(_xy(open_))[:, 1]
    scored["p_win_stage_default"] = scored["stage"].map(stage_p).fillna(0.0)
    scored["ev_model"] = scored["p_win_model"] * scored["acv"]
    scored["ev_stage_default"] = scored["p_win_stage_default"] * scored["acv"]

    # 3. Monte Carlo: each open deal closes (or not) with its model probability.
    #    Deal outcomes alone would give a falsely narrow range, so each run also draws the overall
    #    win rate from its Beta posterior (Beta(won + 1, lost + 1)) and shifts every deal's log-odds
    #    by the same amount. That models "this quarter is better / worse for everyone".
    rng = np.random.default_rng(seed)
    runs = int(cfg["monte_carlo_runs"])
    p = scored["p_win_model"].to_numpy()
    acv = scored["acv"].to_numpy(dtype=float)
    won, lost = int(y.sum()), int(len(y) - y.sum())
    base_logit = np.log(base / (1 - base))
    rates = rng.beta(won + 1, lost + 1, size=runs)
    shifts = np.log(rates / (1 - rates)) - base_logit
    logit_p = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    sims = np.empty(runs)
    batch = 1000
    for start in range(0, runs, batch):
        n = min(batch, runs - start)
        p_run = 1 / (1 + np.exp(-(logit_p[None, :] + shifts[start:start + n, None])))
        sims[start:start + n] = (rng.random((n, len(p))) < p_run) @ acv

    by_rep = (scored.groupby("owner_rep_id")
              .agg(open_deals=("opportunity_id", "count"), pipeline_acv=("acv", "sum"),
                   forecast_model=("ev_model", "sum"), forecast_stage_default=("ev_stage_default", "sum"))
              .reset_index())
    by_rep["overstatement"] = by_rep["forecast_stage_default"] / by_rep["forecast_model"] - 1

    ev_model = float(scored["ev_model"].sum())
    ev_stage = float(scored["ev_stage_default"].sum())
    metrics = {
        "closed_deals_for_training": len(closed),
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_brier": float(brier_score_loss(y, oof)),
        "baseline_brier": float(brier_score_loss(y, np.full_like(oof, base))),
        "brier_skill": 1 - float(brier_score_loss(y, oof)) / float(brier_score_loss(y, np.full_like(oof, base))),
        "calibration_max_gap": float(np.abs(calib["predicted"] - calib["actual"]).max()),
        "historical_win_rate": base,
        "open_deals": len(scored),
        "open_pipeline_acv": float(acv.sum()),
        "forecast_model_expected": ev_model,
        "forecast_stage_default": ev_stage,
        "forecast_historical_rate": base * float(acv.sum()),
        "stage_default_overstatement": ev_stage / ev_model - 1 if ev_model else float("nan"),
        "commit_p10": float(np.percentile(sims, 10)),     # 90% chance to book at least this
        "most_likely_p50": float(np.percentile(sims, 50)),
        "best_case_p90": float(np.percentile(sims, 90)),  # 10% chance to book more than this
        "prob_reaching_stage_default": float((sims >= ev_stage).mean()),
        "avg_stage_default_probability": float(scored["p_win_stage_default"].mean()),
        "avg_model_probability": float(p.mean()),
    }
    return ForecastResult(metrics, calib, scored, sims, by_rep)

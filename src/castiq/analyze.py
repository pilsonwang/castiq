"""Run every analysis on the warehouse and collect the results in one place.

`Results.metrics` is the single source of truth for every number in the dashboard, README, case study,
and resume bullets. Nothing downstream computes its own numbers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from castiq import benchmark, forecast, icp, lead_score, pipeline, quality, routing, win_loss
from castiq import warehouse as wh
from castiq.config import Paths, load_datasets_config, load_icp_config, load_revops_config

log = logging.getLogger(__name__)


@dataclass
class Results:
    metrics: dict[str, Any]
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating | float):
        f = float(obj)
        return None if np.isnan(f) or np.isinf(f) else round(f, 6)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run(paths: Paths, run_benchmark: bool = True) -> Results:
    ds_cfg = load_datasets_config(paths)
    icp_cfg = load_icp_config(paths)
    ops = load_revops_config(paths)
    db = paths.warehouse
    m: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}

    # --- Market, ICP, territories (REAL firmographics) -----------------------------------------
    log.info("market sizing + ICP")
    companies = wh.read(db, "stg_companies")
    scored = icp.score_companies(companies, icp_cfg)
    market = icp.market_sizing(scored, icp_cfg)
    stats_path = paths.raw / "companies" / "country_stats.json"
    country = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    m["market"] = {k: v for k, v in market.items() if k != "by_tier"} | {"country": country}
    frames["market_by_tier"] = market["by_tier"]
    frames["market_by_industry"] = icp.industry_summary(scored)

    tcfg = icp_cfg["territories"]
    strat = tcfg["strategic"]
    is_strat = scored["tier"].isin(strat["tiers"]) & scored["size_band"].isin(strat["size_bands"])
    state_pot = icp.state_summary(scored[~is_strat], tcfg["potential_tiers"])
    frames["state_potential"] = state_pot
    log.info("territory design")
    before = routing.design_territories(state_pot, int(tcfg["count"]), balance=False)
    after = routing.design_territories(state_pot, int(tcfg["count"]), balance=True,
                                       compactness_weight=float(tcfg.get("compactness_weight", 0.05)))
    routed = routing.route_accounts(scored, after, icp_cfg)
    frames["territories_before"] = before.summary
    frames["territories"] = after.summary
    frames["territory_states"] = after.assignment
    frames["territory_states_before"] = before.assignment
    frames["owner_load"] = routing.owner_load(routed)
    m["territories"] = {
        "count": int(tcfg["count"]),
        "cv_before": before.cv, "cv_after": after.cv,
        "max_min_before": before.max_min_ratio, "max_min_after": after.max_min_ratio,
        "strategic_accounts": int(is_strat.sum()),
        "strategic_potential_usd": float(scored.loc[is_strat, "est_annual_spend_usd"].sum()),
        "data_review_accounts": int((routed["route_rule"] == "2-data-review").sum()),
        "named_accounts_per_rep": int(tcfg["rep_capacity_accounts"]),
        "reps": int(tcfg["count"]) * int(tcfg["reps_per_territory"]),
    }
    top_cols = ["name", "industry", "size_band", "city", "state", "website", "fit_score", "tier",
                "est_annual_spend_usd"]
    frames["top_accounts"] = scored[~is_strat].head(250)[top_cols]
    frames["strategic_accounts"] = scored[is_strat].head(100)[top_cols]
    frames["routed_accounts"] = routed

    # --- Pipeline, forecast (CRM) -------------------------------------------------------------
    log.info("pipeline + forecast")
    opps = wh.read(db, "mart_opportunities")
    m["pipeline"] = pipeline.summary(opps, ops["pipeline"])
    seg = pipeline.segment_win_rates(opps, int(ops["pipeline"]["min_segment_n"]))
    for label, t in seg.items():
        frames[f"winrate::{label}"] = t
    m["pipeline"]["significant_segments"] = [
        f"{label}: {row.iloc[0]}" for label, t in seg.items() for _, row in t[t["significant"]].iterrows()
    ]
    reps, chi = pipeline.rep_spread(opps, ops["pipeline"])
    frames["reps"] = reps
    m["reps"] = chi
    frames["stage_mix"] = pipeline.stage_mix(opps)
    frames["funnel"] = wh.read(db, "mart_funnel")
    frames["multithreading"] = pipeline.multithreading_effect(opps)

    fc = forecast.run(opps, ops["forecast"])
    m["forecast"] = fc.metrics
    frames["calibration"] = fc.calibration
    frames["forecast_by_rep"] = fc.by_rep
    arrays = {"forecast_sims": fc.simulation}

    # --- Lead scoring ----------------------------------------------------------------------------
    log.info("lead scoring")
    lcfg = ops["lead_scoring"]
    split = {s: pd.read_parquet(paths.raw / "crm" / f"task_{s}.parquet") for s in ("train", "valid", "test")}
    ls = lead_score.run(split["train"], split["valid"], split["test"], lcfg)
    m["lead_scoring"] = ls.metrics | {"model": ls.model_name, "leakage": ls.leakage_audit,
                                       "tier_used": ds_cfg["crm"]["tier"]}
    frames["lead_tiers"] = ls.tiers
    frames["lead_scorecard"] = ls.scorecard
    frames["lead_models"] = ls.model_comparison
    top2 = ls.tiers.head(2)
    m["lead_scoring"]["top30_share_of_conversions"] = float(top2["share_of_conversions"].sum())
    m["lead_scoring"]["top_tier_lift"] = float(ls.tiers.iloc[0]["lift_vs_avg"])
    m["lead_scoring"]["top_tier_rate"] = float(ls.tiers.iloc[0]["conversion_rate"])
    m["lead_scoring"]["bottom_tier_rate"] = float(ls.tiers.iloc[-1]["conversion_rate"])
    bench_dir = paths.raw / "crm" / "benchmark"
    tiers = ds_cfg["crm"].get("benchmark_tiers", [])
    if run_benchmark and tiers and bench_dir.exists():
        log.info("benchmark across tiers")
        frames["benchmark"] = benchmark.run(bench_dir, tiers, lcfg)

    # --- Win / loss (REAL deals) -----------------------------------------------------------------
    log.info("win/loss")
    deals = wh.read(db, "stg_deals")
    wl = win_loss.run(deals, ops["win_loss"])
    m["win_loss"] = {"deals": wl.n_deals, "base_win_rate": wl.base_win_rate, "cv_auc_mean": wl.cv_auc_mean,
                     "cv_auc_std": wl.cv_auc_std,
                     "q1_win_rate": float(wl.score_quartiles.iloc[0]["win_rate"]),
                     "q4_win_rate": float(wl.score_quartiles.iloc[-1]["win_rate"])}
    frames["win_drivers"] = wl.drivers
    frames["win_quartiles"] = wl.score_quartiles
    frames["checklist"] = wl.checklist

    # --- Data quality ------------------------------------------------------------------------------
    log.info("data quality")
    dq = quality.run(db)
    frames["quality"] = dq
    frames["state_cleanup"] = quality.state_cleanup_examples(db)
    m["quality"] = {"checks": len(dq), "failed_checks": int((dq["status"] == "FAIL").sum()),
                    "seniority_mismatch_pct": float(
                        dq.loc[dq["check"] == "Seniority field contradicts job title", "failed_pct"].iloc[0]),
                    "duplicate_company_websites": int(
                        dq.loc[dq["check"] == "Duplicate company (same website)", "failed"].iloc[0]),
                    "states_fixed": int(frames["state_cleanup"].dropna(subset=["cleaned_to"])["companies"].sum())}

    manifest = paths.raw / "manifest.json"
    m["sources"] = json.loads(manifest.read_text()) if manifest.exists() else {}
    return Results(_jsonable(m), frames, arrays)


def save(results: Results, paths: Paths) -> list[Path]:
    """Write metrics.json and the small, shareable CSVs to docs/; big files go to data/outputs/."""
    out_docs = paths.docs / "data"
    out_docs.mkdir(parents=True, exist_ok=True)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    written = []
    metrics_path = paths.docs / "metrics.json"
    metrics = dict(results.metrics)
    metrics.pop("sources", None)  # sources are in docs/DATA_SOURCES.md and data/raw/manifest.json
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    written.append(metrics_path)
    shareable = ["top_accounts", "strategic_accounts", "territories", "territory_states", "lead_tiers",
                 "lead_scorecard", "win_drivers", "checklist", "quality", "reps", "forecast_by_rep",
                 "benchmark", "market_by_industry", "market_by_tier"]
    for name in shareable:
        if name in results.frames:
            p = out_docs / f"{name}.csv"
            results.frames[name].to_csv(p, index=False)
            written.append(p)
    routed = results.frames.get("routed_accounts")
    if routed is not None:
        p = paths.outputs / "routed_accounts.parquet"
        routed.to_parquet(p, index=False)
        written.append(p)
    return written


def load_metrics(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data

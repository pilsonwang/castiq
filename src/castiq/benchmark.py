"""Benchmark the lead-scoring model on every difficulty tier against the dataset authors' published baselines.

This keeps the main result honest: if the model only worked on the easiest data, this table would show it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from castiq import lead_score


def run(bench_dir: Path, tiers: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    quick = dict(cfg, bootstrap_runs=min(int(cfg["bootstrap_runs"]), 300))
    for tier in tiers:
        split = {s: pd.read_parquet(bench_dir / f"{tier}_{s}.parquet") for s in ("train", "valid", "test")}
        published = json.loads((bench_dir / f"{tier}_published_metrics.json").read_text())
        res = lead_score.run(split["train"], split["valid"], split["test"], quick)
        med, spread = published["medians"], published["spreads_max_minus_min"]
        knobs = published.get("difficulty_knobs", {})
        rows.append({
            "tier": tier,
            "missing_rate": knobs.get("missing_rate"),
            "noise_scale": knobs.get("noise_scale"),
            "base_rate": res.metrics["test_base_rate"],
            "model": res.model_name,
            "our_test_auc": res.metrics["test_auc"],
            "our_auc_ci_low": res.metrics["test_auc_ci_low"],
            "our_auc_ci_high": res.metrics["test_auc_ci_high"],
            "published_lr_auc_median": med["lr_auc"],
            "published_gbm_auc_median": med["gbm_auc"],
            "published_auc_spread": max(spread.get("lr_auc", 0), spread.get("gbm_auc", 0)),
            "top_tier_lift": float(res.tiers.iloc[0]["lift_vs_avg"]),
            "leak_auc_inflation": res.leakage_audit["auc_inflation"],
        })
    return pd.DataFrame(rows)

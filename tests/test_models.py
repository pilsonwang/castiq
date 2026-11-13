from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from castiq import forecast, lead_score, pipeline, win_loss
from castiq import warehouse as wh
from castiq.config import load_revops_config
from tests.conftest import make_deals, make_task_split


@pytest.fixture()
def ops(project):  # type: ignore[no-untyped-def]
    return load_revops_config(project)


# ---------------------------------------------------------------- lead scoring
def test_leakage_column_is_never_a_feature(ops) -> None:  # type: ignore[no-untyped-def]
    df = make_task_split(np.random.default_rng(0), 50)
    num, cat = lead_score.split_features(df, ops["lead_scoring"])
    assert "total_touches_all" not in num + cat
    assert "lead_id" not in num + cat and "converted_within_90_days" not in num + cat
    num_l, cat_l = lead_score.split_features(df, ops["lead_scoring"], include_leaky=True)
    assert "total_touches_all" in num_l + cat_l
    assert {"industry", "seniority", "has_open_opportunity"} <= set(cat)
    assert {"touch_count", "session_count", "days_since_last_touch"} <= set(num)


def test_split_features_handles_every_string_dtype(ops) -> None:  # type: ignore[no-untyped-def]
    df = pd.DataFrame({"a": pd.Series(["x", "y"], dtype="string"), "b": pd.Series(["x", "y"], dtype=object),
                       "c": pd.Series(["x", "y"], dtype="category"), "d": [True, False], "e": [1.0, 2.0],
                       "converted_within_90_days": [True, False]})
    num, cat = lead_score.split_features(df, ops["lead_scoring"])
    assert num == ["e"] and cat == ["a", "b", "c", "d"]


def test_lead_score_run_learns_signal_and_measures_leak(ops) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(1)
    cfg = dict(ops["lead_scoring"], bootstrap_runs=100)
    res = lead_score.run(make_task_split(rng, 800), make_task_split(rng, 300), make_task_split(rng, 400), cfg)
    m = res.metrics
    assert 0.6 < m["test_auc"] <= 1.0
    assert m["test_auc_ci_low"] < m["test_auc"] < m["test_auc_ci_high"]
    assert res.leakage_audit["auc_inflation"] > 0.05  # the planted leak is strong
    assert res.tiers["leads"].sum() == 400
    assert res.tiers["share_of_conversions"].sum() == pytest.approx(1.0)
    assert res.tiers.iloc[0]["conversion_rate"] > res.tiers.iloc[-1]["conversion_rate"]
    assert not res.scorecard["feature"].str.contains("__").any()


def test_tier_table_cuts_by_share() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    p = np.linspace(1, 0, 10)
    t = lead_score.tier_table(y, p, {"A": 0.2, "B": 0.3, "C": 0.5})
    assert t["leads"].tolist() == [2, 3, 5]
    assert t["conversions"].tolist() == [2, 0, 1]


@pytest.mark.parametrize("name,expected", [
    ("cat__industry_manufacturing", "industry = manufacturing"),
    ("cat__has_open_opportunity_True", "has open opportunity = yes"),
    ("num__touch_count", "touch count"),
    ("num__missingindicator_days_since_last_touch", "days since last touch is missing"),
])
def test_readable_feature(name: str, expected: str) -> None:
    assert lead_score.readable_feature(name, ["industry", "has_open_opportunity"]) == expected


# ---------------------------------------------------------------- win/loss
def test_win_loss_finds_the_planted_driver(ops) -> None:  # type: ignore[no-untyped-def]
    deals = make_deals(np.random.default_rng(2), 300)
    deals["is_won"] = deals["Status"] == "Won"
    res = win_loss.run(deals, dict(ops["win_loss"], cv_repeats=2))
    assert res.cv_auc_mean > 0.65
    top = res.drivers.iloc[0]
    assert top["field"] == "Client"
    assert set(res.checklist["n"]) and (res.checklist["n"] >= ops["win_loss"]["min_group_n"]).all()
    assert res.score_quartiles["win_rate"].iloc[-1] > res.score_quartiles["win_rate"].iloc[0]


# ---------------------------------------------------------------- pipeline + forecast
def test_pipeline_summary_matches_raw_counts(project, ops) -> None:  # type: ignore[no-untyped-def]
    opps = wh.read(project.warehouse, "mart_opportunities")
    raw = pd.read_parquet(project.raw / "crm" / "opportunities.parquet")
    s = pipeline.summary(opps, ops["pipeline"])
    assert s["opportunities"] == len(raw)
    assert s["won"] == int((raw["stage"] == "closed_won").sum())
    assert s["closed_opportunities"] == int(raw["stage"].isin(["closed_won", "closed_lost"]).sum())
    assert s["win_rate_ci"][0] <= s["win_rate"] <= s["win_rate_ci"][1]
    reps, chi = pipeline.rep_spread(opps, ops["pipeline"])
    assert len(reps) == 4 and 0 <= chi["p_value"] <= 1


def test_forecast_ranges_and_default_overstatement(project, ops) -> None:  # type: ignore[no-untyped-def]
    opps = wh.read(project.warehouse, "mart_opportunities")
    res = forecast.run(opps, ops["forecast"])
    m = res.metrics
    assert m["commit_p10"] <= m["most_likely_p50"] <= m["best_case_p90"]
    assert m["commit_p10"] <= m["open_pipeline_acv"]
    # fixture win rate is ~50% while defaults are 50-90% -> defaults overstate
    assert m["forecast_stage_default"] > m["forecast_model_expected"]
    assert len(res.simulation) == ops["forecast"]["monte_carlo_runs"]
    assert res.open_deals["p_win_model"].between(0, 1).all()

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from castiq import icp, routing
from castiq.config import load_icp_config


@pytest.fixture()
def cfg(project):  # type: ignore[no-untyped-def]
    return load_icp_config(project)


def _companies(rows: list[tuple[str, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame([{"handle": f"h{i}", "name": f"n{i}", "industry": ind, "size_band": size, "state": st}
                         for i, (ind, size, st) in enumerate(rows)])


def test_fit_score_is_the_documented_weighted_sum(cfg) -> None:  # type: ignore[no-untyped-def]
    df = icp.score_companies(_companies([("motor vehicle manufacturing", "501-1K", "Michigan")]), cfg)
    # 100 * (0.5*1.0 + 0.35*1.0 + 0.15*1.0)
    assert df.loc[0, "fit_score"] == 100.0
    assert df.loc[0, "tier"] == "A"
    df = icp.score_companies(_companies([("manufacturing", "51-200", "Nevada")]), cfg)
    # 100 * (0.5*0.5 + 0.35*0.6 + 0.15*0.7) = 56.5 -> tier C
    assert df.loc[0, "fit_score"] == pytest.approx(56.5)
    assert df.loc[0, "tier"] == "C"


def test_unknown_state_and_size_get_penalised(cfg) -> None:  # type: ignore[no-untyped-def]
    df = icp.score_companies(_companies([("automotive", None, None)]), cfg)  # type: ignore[list-item]
    expected = 100 * (0.5 * 1.0 + 0.35 * 0.2 + 0.15 * 0.35)
    assert df.loc[0, "fit_score"] == pytest.approx(expected, abs=0.05)
    assert df.loc[0, "size_band"] == "unknown"


def test_out_of_market_industries_are_dropped(cfg) -> None:  # type: ignore[no-untyped-def]
    df = icp.score_companies(_companies([("bakery", "1K-5K", "Ohio"), ("automotive", "1K-5K", "Ohio")]), cfg)
    assert df["industry"].tolist() == ["automotive"]


def test_scoring_order_is_deterministic(cfg) -> None:  # type: ignore[no-untyped-def]
    rows = [("automotive", "501-1K", "Ohio")] * 5 + [("machinery manufacturing", "1K-5K", "Texas")] * 5
    df = _companies(rows)
    a = icp.score_companies(df, cfg)["handle"].tolist()
    b = icp.score_companies(df.sample(frac=1, random_state=3), cfg)["handle"].tolist()
    assert a == b


def test_market_sizing_adds_up(cfg) -> None:  # type: ignore[no-untyped-def]
    df = icp.score_companies(_companies([("automotive", "1K-5K", "Ohio"), ("manufacturing", "11-50", "Utah")]), cfg)
    m = icp.market_sizing(df, cfg)
    assert m["tam_usd"] == pytest.approx(df["est_annual_spend_usd"].sum())
    assert m["som_usd"] == pytest.approx(m["sam_usd"] * cfg["obtainable_share"])
    assert m["by_tier"]["accounts"].sum() == len(df)


def _state_potential() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    states = ["Michigan", "Ohio", "Indiana", "Illinois", "California", "Texas", "New York", "Georgia",
              "Pennsylvania", "Wisconsin", "Tennessee", "Washington", "Florida", "North Carolina"]
    return pd.DataFrame({"state": states, "potential_usd": rng.uniform(1e8, 1e9, len(states)),
                         "accounts": rng.integers(10, 100, len(states))})


@pytest.mark.parametrize("k", [3, 4, 6])
def test_territories_are_contiguous_and_cover_every_state(k: int) -> None:
    plan = routing.design_territories(_state_potential(), k)
    a = plan.assignment
    assert a["territory"].nunique() == k
    assert len(a) == len(set(a["state"]))
    states = a["state"].tolist()
    nbrs = routing._neighbour_index(states)
    labels = a["territory"].to_numpy()
    for t in range(k):
        assert routing.is_contiguous(labels, nbrs, t), f"territory {t} is split"


def test_balancing_improves_on_geography_only() -> None:
    before = routing.design_territories(_state_potential(), 4, balance=False)
    after = routing.design_territories(_state_potential(), 4, balance=True)
    assert after.cv < before.cv
    assert after.summary["potential_usd"].sum() == pytest.approx(before.summary["potential_usd"].sum())


def test_route_accounts_rules(cfg) -> None:  # type: ignore[no-untyped-def]
    rows = [("automotive", "10K+", "Michigan")]  # strategic
    rows += [("automotive", None, None)]  # no state -> data review (tier must be A/B to be routed)
    rows += [("automotive", "501-1K", "Ohio")] * 30  # territory -> reps then overflow
    df = icp.score_companies(_companies(rows), cfg)  # type: ignore[arg-type]
    df.loc[df["state"].isna(), "tier"] = "B"  # force the no-state account into the routed set
    plan = routing.design_territories(_state_potential(), 3)
    routed = routing.route_accounts(df, plan, cfg)
    by_rule = routed.groupby("route_rule").size()
    assert by_rule["1-strategic"] == 1
    assert by_rule["2-data-review"] == 1
    terr = routed[routed["route_rule"] == "3-territory"]
    cap = cfg["territories"]["rep_capacity_accounts"] * cfg["territories"]["reps_per_territory"]
    assert (~terr["owner"].str.contains("Inside sales")).sum() == min(30, cap)
    assert (terr["owner"].str.contains("Inside sales")).sum() == max(0, 30 - cap)
    load = routing.owner_load(routed)
    assert load["accounts"].sum() == len(routed)

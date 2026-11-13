from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from castiq import geo
from castiq.config import validate_icp_config
from castiq.stats import coefficient_of_variation, gini, rate_table, wilson_interval


# ---------------------------------------------------------------- geo
@pytest.mark.parametrize("raw,expected", [
    ("Michigan", "Michigan"), ("  michigan ", "Michigan"), ("MI", "Michigan"), ("Tx Texas", "Texas"),
    ("York", "New York"), ("D.C.", "District of Columbia"), ("District Of Columbia", "District of Columbia"),
    ("Flrida", "Florida"), ("Pensylvania", "Pennsylvania"), ("West Virginia", "West Virginia"),
    ("Arkansas", "Arkansas"), ("Kansas", "Kansas"),
    ("Ontario", None), ("Guangdong", None), ("Portland", None), ("Null", None), ("", None), (None, None), (42, None),
])
def test_normalize_state(raw: object, expected: str | None) -> None:
    assert geo.normalize_state(raw) == expected


def test_adjacency_is_symmetric_and_complete() -> None:
    assert set(geo.ADJACENCY) == set(geo.STATES)
    for s, nbrs in geo.ADJACENCY.items():
        assert nbrs, s
        for n in nbrs:
            assert s in geo.ADJACENCY[n], f"{s} -> {n} is not symmetric"


def test_tiles_are_unique() -> None:
    tiles = [geo.tile(s) for s in geo.STATES]
    assert len(tiles) == len(set(tiles)) == 51


# ---------------------------------------------------------------- stats
def test_wilson_interval_known_value() -> None:
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_edges() -> None:
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    assert all(math.isnan(v) for v in wilson_interval(0, 0))
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_rate_table_flags_significant_groups() -> None:
    df = pd.DataFrame({"g": ["a"] * 200 + ["b"] * 200 + ["c"] * 5,
                       "y": [1] * 160 + [0] * 40 + [1] * 40 + [0] * 160 + [1, 0, 1, 0, 1]})
    t = rate_table(df, "g", "y", min_n=10).set_index("g")
    assert "c" not in t.index  # below min_n
    assert t.loc["a", "rate"] == pytest.approx(0.8)
    assert bool(t.loc["a", "significant"]) and bool(t.loc["b", "significant"])
    assert t.loc["a", "lift_pts"] > 0 > t.loc["b", "lift_pts"]


def test_cv_and_gini() -> None:
    assert coefficient_of_variation(np.array([5, 5, 5])) == 0
    assert gini(np.array([1, 1, 1, 1])) == pytest.approx(0)
    assert gini(np.array([0, 0, 0, 10])) == pytest.approx(0.75)


# ---------------------------------------------------------------- config validation
def _icp() -> dict:
    return {"weights": {"industry": 0.5, "size": 0.35, "geo": 0.15}, "industry_fit": {"a": 1.0},
            "size_fit": {"1-10": 0.1}, "tiers": {"A": 80, "B": 65, "C": 50}}


def test_icp_config_valid() -> None:
    validate_icp_config(_icp())


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: c["weights"].update(geo=0.5), "add up to 1.0"),
    (lambda c: c["industry_fit"].update(a=1.5), "between 0 and 1"),
    (lambda c: c["tiers"].update(B=90), "A > B > C"),
])
def test_icp_config_errors(mutate, msg: str) -> None:  # type: ignore[no-untyped-def]
    cfg = _icp()
    mutate(cfg)
    with pytest.raises(ValueError, match=msg):
        validate_icp_config(cfg)

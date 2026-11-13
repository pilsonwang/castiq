from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from castiq import ingest, quality
from castiq import warehouse as wh


def test_warehouse_builds_every_layer(project) -> None:  # type: ignore[no-untyped-def]
    tables = set(wh.read(project.warehouse, "select table_name from information_schema.tables")["table_name"])
    for t in ("stg_accounts", "stg_deals", "stg_companies", "int_lead_engagement", "int_lead_snapshot",
              "mart_opportunities", "mart_funnel", "mart_rep_scorecard"):
        assert t in tables


def test_each_deal_is_measured_at_its_own_snapshot(project) -> None:  # type: ignore[no-untyped-def]
    df = wh.read(project.warehouse, "select lead_created_at, as_of_date from mart_opportunities")
    delta = (pd.to_datetime(df["as_of_date"]) - pd.to_datetime(df["lead_created_at"])).dt.days
    assert (delta == 30).all()


def test_funnel_is_monotonic(project) -> None:  # type: ignore[no-untyped-def]
    f = wh.read(project.warehouse, "mart_funnel").sort_values("step_order")
    assert f.loc[f["step"] == "Closed won", "n"].iloc[0] <= f.loc[f["step"] == "Closed (won or lost)", "n"].iloc[0]


@pytest.mark.parametrize("title,level", [("CFO", "c_suite"), ("VP Finance", "vp"), ("Head of IT", "director"),
                                         ("AP Lead", "manager"), ("Analyst", None), (None, None)])
def test_title_seniority(title: str | None, level: str | None) -> None:
    assert quality.title_seniority(title) == level


def test_quality_checks_catch_planted_defects(project) -> None:  # type: ignore[no-untyped-def]
    dq = quality.run(project.warehouse).set_index("check")
    assert dq.loc["Seniority field contradicts job title", "failed"] == 25
    assert dq.loc["Contact with a personal email domain", "failed"] == 30
    assert dq.loc["Open deal with no logged sales activity", "failed"] == 10
    assert dq.loc["Duplicate company (same website)", "failed"] == 3
    assert dq.loc["Company state could not be mapped", "failed"] == 2
    assert dq.loc["Company state missing", "failed"] == 2
    assert dq.loc["Company size band missing", "failed"] == 1
    assert dq.loc["Duplicate opportunity IDs", "status"] == "PASS"
    assert set(dq["severity"]) <= {"high", "medium", "low"}
    fixed = quality.state_cleanup_examples(project.warehouse)
    assert {"Flrida", "York", "Tx Texas"} <= set(fixed["state_raw"])


def test_filter_companies_keeps_country_and_market(companies_csv_gz: Path) -> None:
    df, stats = ingest.filter_companies(companies_csv_gz, "US", ["Motor Vehicle Manufacturing",
                                                                 "machinery manufacturing", "manufacturing"])
    assert stats["country_companies"] == 401  # 400 fixture rows + 1 US bakery; the GB row is excluded
    assert len(df) == 400  # the bakery is out of market
    assert set(df["country_code"]) == {"US"}
    assert df.loc[df["state_raw"] == "Flrida", "state"].iloc[0] == "Florida"


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("CASTIQ_NETWORK_TESTS") != "1", reason="set CASTIQ_NETWORK_TESTS=1")
def test_pinned_deals_file_matches_manifest(tmp_path: Path) -> None:
    """Downloads the small real deals file from Hugging Face and checks size and label balance."""
    from castiq.config import PROJECT_ROOT, Paths, load_datasets_config

    paths = Paths(root=tmp_path, config_dir=PROJECT_ROOT / "config").ensure()
    entries = ingest.fetch_deals(paths, load_datasets_config(paths)["deals"])
    assert entries[0]["rows"] == 448
    df = pd.read_parquet(tmp_path / "data" / "raw" / "deals" / "deals.parquet")
    assert set(df["Status"]) == {"Won", "Lost"}
    manifest = PROJECT_ROOT / "data" / "raw" / "manifest.json"
    if manifest.exists():
        pinned = json.loads(manifest.read_text())["deals"][0]["sha256"]
        assert entries[0]["sha256"] == pinned

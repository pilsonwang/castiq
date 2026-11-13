"""Test fixtures: a tiny, fully offline CastIQ project with known, planted data problems.

The fixture data is generated here for tests only. It mirrors the real schemas so the same code paths run,
and it plants specific defects (wrong seniority, duplicate websites, typo states...) so tests can assert
exact counts.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from castiq.config import PROJECT_ROOT, Paths

RNG_SEED = 7
N_LEADS = 400


def _crm_tables(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    n_acc, n_con = 120, 300
    industries = ["manufacturing", "logistics", "professional_services", "healthcare_non_clinical"]
    accounts = pd.DataFrame({
        "account_id": [f"acct_{i:04d}" for i in range(n_acc)],
        "company_name": [f"Company {i}" for i in range(n_acc)],
        "industry": rng.choice(industries, n_acc),
        "region": rng.choice(["US", "UK"], n_acc),
        "employee_band": rng.choice(["200-499", "500-999", "1000-1999", "2000+"], n_acc),
        "estimated_revenue_band": rng.choice(["$1M-$10M", "$10M-$50M", "$50M-$200M"], n_acc),
        "process_maturity_band": rng.choice(["low", "medium", "high"], n_acc),
        "created_at": "2023-01-01",
    })
    titles = [("CFO", "c_suite"), ("VP Finance", "vp"), ("Head of IT", "director"), ("Procurement Manager", "manager")]
    picks = rng.integers(0, len(titles), n_con)
    seniority = [titles[p][1] for p in picks]
    # PLANT: the first 25 contacts get a seniority that contradicts their title.
    for i in range(25):
        seniority[i] = "individual_contributor"
    contacts = pd.DataFrame({
        "contact_id": [f"cnt_{i:04d}" for i in range(n_con)],
        "account_id": [f"acct_{i % n_acc:04d}" for i in range(n_con)],
        "job_title": [titles[p][0] for p in picks],
        "role_function": rng.choice(["vp_finance", "it_director", "procurement_manager", "ap_manager"], n_con),
        "seniority": seniority,
        "buyer_role": rng.choice(["economic_buyer", "champion", "end_user", "technical_evaluator"], n_con),
        # PLANT: exactly 30 personal email domains.
        "email_domain_type": ["personal"] * 30 + ["corporate"] * (n_con - 30),
        "created_at": "2023-06-01",
    })
    lead_contacts = rng.integers(0, n_con, N_LEADS)
    created = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 30, N_LEADS), unit="D")
    leads = pd.DataFrame({
        "lead_id": [f"lead_{i:04d}" for i in range(N_LEADS)],
        "contact_id": [f"cnt_{c:04d}" for c in lead_contacts],
        "account_id": [f"acct_{c % n_acc:04d}" for c in lead_contacts],
        "lead_created_at": created.strftime("%Y-%m-%d"),
        "lead_source": rng.choice(["inbound_marketing", "sdr_outbound", "partner_referral"], N_LEADS),
        "first_touch_channel": "email",
        "owner_rep_id": [f"rep_{i % 4 + 1:03d}" for i in range(N_LEADS)],
    })
    # Opportunities for 300 leads; ~half closed. Manufacturing wins more (a real signal for the models).
    opp_leads = np.arange(300)
    industry_by_acc = accounts.set_index("account_id")["industry"]
    is_mfg = industry_by_acc.loc[leads.loc[opp_leads, "account_id"]].to_numpy() == "manufacturing"
    stage = []
    for i, m in zip(opp_leads, is_mfg, strict=True):
        if i < 150:
            stage.append("closed_won" if rng.random() < (0.7 if m else 0.35) else "closed_lost")
        else:
            stage.append(rng.choice(["negotiation", "proposal_sent", "demo_completed"]))
    opps = pd.DataFrame({
        "opportunity_id": [f"opp_{i:04d}" for i in opp_leads],
        "lead_id": [f"lead_{i:04d}" for i in opp_leads],
        "created_at": (created[opp_leads] + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
        "stage": stage,
        "estimated_acv": pd.array(rng.integers(20_000, 120_000, len(opp_leads)), dtype="Int64"),
    })
    # Activities for every lead except leads 290-299 (PLANT: 10 open deals with no activity).
    act_leads = [i for i in range(N_LEADS) if not 290 <= i < 300]
    acts = []
    for k, i in enumerate(act_leads):
        for j in range(2):
            acts.append({"activity_id": f"act_{k}_{j}", "lead_id": f"lead_{i:04d}", "rep_id": f"rep_{i % 4 + 1:03d}",
                         "activity_timestamp": (created[i] + pd.Timedelta(days=1 + j * 2)).strftime("%Y-%m-%d"),
                         "activity_type": rng.choice(["call", "email", "meeting", "demo"]),
                         "activity_outcome": rng.choice(["connected", "meeting_set", "no_answer", "demo_completed"])})
    activities = pd.DataFrame(acts)
    sessions = pd.DataFrame({
        "session_id": [f"s_{i}" for i in range(N_LEADS)], "lead_id": leads["lead_id"],
        "session_timestamp": leads["lead_created_at"], "session_type": "website",
        "page_views": pd.array(rng.integers(1, 9, N_LEADS), dtype="Int64"),
        "pricing_page_views": pd.array(rng.integers(0, 3, N_LEADS), dtype="Int64"),
        "demo_page_views": pd.array(rng.integers(0, 2, N_LEADS), dtype="Int64"),
        "session_duration_seconds": pd.array(rng.integers(30, 600, N_LEADS), dtype="Int64"),
    })
    touches = pd.DataFrame({
        "touch_id": [f"t_{i}" for i in range(N_LEADS)], "lead_id": leads["lead_id"],
        "touch_timestamp": leads["lead_created_at"], "touch_type": "email",
        "touch_channel": leads["lead_source"],
        "touch_direction": rng.choice(["inbound", "outbound"], N_LEADS), "campaign_id": None,
    })
    return {"accounts": accounts, "contacts": contacts, "leads": leads, "opportunities": opps,
            "sales_activities": activities, "sessions": sessions, "touches": touches}


def make_task_split(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Lead-scoring task frame with real column names, a learnable signal, and a leaky column."""
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=1.0, size=n)) > 0.9
    return pd.DataFrame({
        "account_id": [f"a{i}" for i in range(n)], "contact_id": [f"c{i}" for i in range(n)],
        "lead_id": [f"l{i}" for i in range(n)], "lead_created_at": "2024-01-01",
        "industry": rng.choice(["manufacturing", "logistics"], n),
        "seniority": rng.choice(["vp", "manager", None], n),
        "has_open_opportunity": pd.array(signal > 0.3, dtype="boolean"),
        "touch_count": pd.array((signal * 2 + 5).round().clip(0), dtype="Int64"),
        "session_count": pd.array(rng.integers(0, 5, n), dtype="Int64"),
        "days_since_last_touch": rng.uniform(0, 30, n),
        # Leaky: counts the future, so it carries the label almost directly.
        "total_touches_all": pd.array((y * 6 + rng.integers(0, 3, n)), dtype="Int64"),
        "converted_within_90_days": pd.array(y, dtype="boolean"),
    })


def make_deals(rng: np.random.Generator, n: int = 240) -> pd.DataFrame:
    from castiq.win_loss import FIELDS

    df = pd.DataFrame({c: rng.choice(["Yes", "No"], n) for c in FIELDS})
    df["Client"] = rng.choice(["Current", "New"], n)
    p = np.where(df["Client"] == "Current", 0.75, 0.2)
    df["Status"] = np.where(rng.random(n) < p, "Won", "Lost")
    df["Product"] = "Product A"
    df["Seller"] = "Seller 1"
    return df


def make_companies() -> pd.DataFrame:
    rows = []
    states = ["Michigan", "Ohio", "Texas", "California", "New York", "Georgia", "Washington", "Florida",
              "Illinois", "Pennsylvania"]
    for i in range(400):
        s = states[i % len(states)]
        rows.append({"handle": f"company/c{i}", "name": f"Co {i}", "website": f"c{i}.com",
                     "industry": ["motor vehicle manufacturing", "machinery manufacturing", "manufacturing"][i % 3],
                     "size": ["51-200", "201-500", "501-1K", "1K-5K", "10K+"][i % 5],
                     "type": "Privately Held", "founded": 1990, "city": "Town", "state": s, "state_raw": s,
                     "country_code": "US"})
    # PLANT: 3 typo/alias states that clean, 2 that cannot be mapped, 2 missing, 4 duplicate websites.
    rows[0]["state_raw"], rows[0]["state"] = "Flrida", "Florida"
    rows[1]["state_raw"], rows[1]["state"] = "York", "New York"
    rows[2]["state_raw"], rows[2]["state"] = "Tx Texas", "Texas"
    rows[3]["state_raw"], rows[3]["state"] = "Ontario", None
    rows[4]["state_raw"], rows[4]["state"] = "Guangdong", None
    rows[5]["state_raw"], rows[5]["state"] = None, None
    rows[6]["state_raw"], rows[6]["state"] = None, None
    for i in range(10, 14):
        rows[i]["website"] = "dupe.com"  # 4 rows share one site -> 3 duplicates
    rows[20]["size"] = None
    return pd.DataFrame(rows)


def build_fixture_project(root: Path) -> Paths:
    rng = np.random.default_rng(RNG_SEED)
    shutil.copytree(PROJECT_ROOT / "config", root / "config")
    paths = Paths(root=root, config_dir=root / "config").ensure()
    crm = paths.raw / "crm"
    crm.mkdir(parents=True)
    for name, df in _crm_tables(rng).items():
        df.to_parquet(crm / f"{name}.parquet", index=False)
    for split, n in (("train", 600), ("valid", 200), ("test", 300)):
        make_task_split(rng, n).to_parquet(crm / f"task_{split}.parquet", index=False)
    bench = crm / "benchmark"
    bench.mkdir()
    for split, n in (("train", 600), ("valid", 200), ("test", 300)):
        make_task_split(rng, n).to_parquet(bench / f"intro_{split}.parquet", index=False)
    (bench / "intro_published_metrics.json").write_text(json.dumps({
        "medians": {"lr_auc": 0.7, "gbm_auc": 0.68}, "spreads_max_minus_min": {"lr_auc": 0.05, "gbm_auc": 0.06},
        "difficulty_knobs": {"missing_rate": 0.02, "noise_scale": 0.1}}))
    (paths.raw / "deals").mkdir()
    make_deals(rng).to_parquet(paths.raw / "deals" / "deals.parquet", index=False)
    (paths.raw / "companies").mkdir()
    make_companies().to_parquet(paths.raw / "companies" / "companies.parquet", index=False)
    (paths.raw / "companies" / "country_stats.json").write_text(json.dumps({"country_companies": 5000}))
    # Benchmark only the fixture tier, smaller runs, fewer territories (fixture has 10 states).
    ds = (paths.config_dir / "datasets.yaml").read_text().replace(
        "benchmark_tiers: [intro, intermediate, advanced]", "benchmark_tiers: [intro]")
    (paths.config_dir / "datasets.yaml").write_text(ds)
    rv = (paths.config_dir / "revops.yaml").read_text()
    rv = rv.replace("monte_carlo_runs: 20000", "monte_carlo_runs: 2000").replace("bootstrap_runs: 1000",
                                                                                 "bootstrap_runs: 100")
    rv = rv.replace("cv_repeats: 20", "cv_repeats: 2")
    (paths.config_dir / "revops.yaml").write_text(rv)
    icp = (paths.config_dir / "icp.yaml").read_text().replace("count: 6 ", "count: 4 ")
    icp = icp.replace("rep_capacity_accounts: 150", "rep_capacity_accounts: 10")
    (paths.config_dir / "icp.yaml").write_text(icp)
    return paths


@pytest.fixture(scope="session")
def project(tmp_path_factory: pytest.TempPathFactory) -> Paths:
    """A fixture project with raw data and a built warehouse."""
    from castiq import warehouse

    root = tmp_path_factory.mktemp("castiq_project")
    paths = build_fixture_project(root)
    warehouse.build(paths.raw, paths.warehouse, snapshot_day=30)
    return paths


@pytest.fixture()
def companies_csv_gz(tmp_path: Path) -> Path:
    df = make_companies().drop(columns=["state_raw"])
    df.loc[0, "state"] = "Flrida"
    extra = pd.DataFrame([{"handle": "company/x", "name": "Bakery", "website": "b.com", "industry": "Bakery",
                           "size": "11-50", "type": "", "founded": "", "city": "", "state": "Ohio",
                           "country_code": "US"},
                          {"handle": "company/y", "name": "UK Co", "website": "u.com",
                           "industry": "Machinery Manufacturing", "size": "51-200", "type": "", "founded": "",
                           "city": "", "state": "Kent", "country_code": "GB"}])
    out = tmp_path / "companies.csv.gz"
    with gzip.open(out, "wt") as fh:
        pd.concat([df, extra]).to_csv(fh, index=False)
    return out

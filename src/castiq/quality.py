"""CRM and firmographic hygiene checks. RevOps owns data quality - this is the weekly report.

Each check returns: what was tested, how many rows failed, why it matters, and who fixes it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


@dataclass(frozen=True)
class Check:
    name: str
    table: str
    severity: str  # high / medium / low
    why: str
    owner: str
    failed_sql: str  # SELECT count(*) of failing rows
    total_sql: str


SENIORITY_RULES: list[tuple[str, str]] = [
    (r"\b(cfo|cto|ceo|coo|chief)\b", "c_suite"),
    (r"\b(vp|vice president)\b", "vp"),
    (r"\b(head|director)\b", "director"),
    (r"\b(manager|lead)\b", "manager"),
]


def title_seniority(title: str | None) -> str | None:
    """Seniority implied by a job title, or None when the title gives no clear signal."""
    if not title:
        return None
    t = title.lower()
    for pattern, level in SENIORITY_RULES:
        if re.search(pattern, t):
            return level
    return None


CHECKS: list[Check] = [
    Check("Duplicate opportunity IDs", "opportunities", "high",
          "Double-counts pipeline and bookings.", "RevOps",
          "select count(*) - count(distinct opportunity_id) from stg_opportunities",
          "select count(*) from stg_opportunities"),
    Check("Opportunity without a lead", "opportunities", "high",
          "Breaks source attribution and marketing ROI.", "RevOps",
          "select count(*) from stg_opportunities o left join stg_leads l using (lead_id) where l.lead_id is null",
          "select count(*) from stg_opportunities"),
    Check("Lead account differs from its contact's account", "leads", "high",
          "Activity and pipeline roll up to the wrong company.", "RevOps / SDR",
          "select count(*) from stg_leads l join stg_contacts c using (contact_id) where l.account_id <> c.account_id",
          "select count(*) from stg_leads"),
    Check("Opportunity created before its lead", "opportunities", "medium",
          "Impossible order of events: a sign of back-dated or merged records.", "RevOps",
          "select count(*) from stg_opportunities o join stg_leads l using (lead_id) "
          "where o.opp_created_at < l.lead_created_at",
          "select count(*) from stg_opportunities"),
    Check("Open deal with no logged sales activity", "opportunities", "high",
          "No activity = no next step. These deals inflate the forecast.", "Sales manager",
          "select count(*) from stg_opportunities o left join (select distinct lead_id from stg_activities) a "
          "using (lead_id) where not o.is_closed and a.lead_id is null",
          "select count(*) from stg_opportunities where not is_closed"),
    Check("Activity logged by a rep who does not own the lead", "activities", "low",
          "Can be fine (team selling), but breaks rep-level activity metrics.", "Sales manager",
          "select count(*) from stg_activities a join stg_leads l using (lead_id) where a.rep_id <> l.owner_rep_id",
          "select count(*) from stg_activities"),
    Check("Contact with a personal email domain", "contacts", "medium",
          "Personal emails bounce off enrichment and break account matching.", "SDR",
          "select count(*) from stg_contacts where email_domain_type = 'personal'",
          "select count(*) from stg_contacts"),
    Check("Opportunity amount missing or zero", "opportunities", "high",
          "Unforecastable pipeline.", "Account executive",
          "select count(*) from stg_opportunities where acv is null or acv <= 0",
          "select count(*) from stg_opportunities"),
    Check("Company state could not be mapped", "companies", "medium",
          "Account cannot be routed to a territory.", "RevOps",
          "select count(*) from stg_companies where state_raw is not null and state is null",
          "select count(*) from stg_companies"),
    Check("Company state missing", "companies", "medium",
          "Account cannot be routed to a territory.", "RevOps / enrichment",
          "select count(*) from stg_companies where state_raw is null",
          "select count(*) from stg_companies"),
    Check("Company size band missing", "companies", "medium",
          "ICP fit score falls back to a low default.", "Enrichment vendor",
          "select count(*) from stg_companies where size_band = 'unknown'",
          "select count(*) from stg_companies"),
    Check("Company website missing", "companies", "low",
          "No domain = no email enrichment or intent matching.", "Enrichment vendor",
          "select count(*) from stg_companies where website is null or website = ''",
          "select count(*) from stg_companies"),
    Check("Duplicate company (same website)", "companies", "medium",
          "Two reps can work the same account.", "RevOps",
          "select coalesce(sum(n - 1), 0) from (select website, count(*) n from stg_companies "
          "where website is not null and website <> '' group by 1 having count(*) > 1)",
          "select count(*) from stg_companies"),
]


def _python_checks(con: duckdb.DuckDBPyConnection) -> list[dict]:
    contacts = con.execute("select job_title, seniority from stg_contacts").df()
    implied = contacts["job_title"].map(title_seniority)
    known = implied.notna()
    mismatch = int((implied[known] != contacts.loc[known, "seniority"]).sum())
    return [{
        "check": "Seniority field contradicts job title", "table": "contacts", "severity": "medium",
        "why": "e.g. a 'CFO' stored as individual contributor - persona routing and scoring go wrong.",
        "owner": "SDR / enrichment", "failed": mismatch, "total": int(known.sum()),
    }]


def run(db_path: Path, extra: list[Callable[[duckdb.DuckDBPyConnection], list[dict]]] | None = None
        ) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = []
        for c in CHECKS:
            failed = con.execute(c.failed_sql).fetchall()[0][0]
            total = con.execute(c.total_sql).fetchall()[0][0]
            rows.append({"check": c.name, "table": c.table, "severity": c.severity, "why": c.why,
                         "owner": c.owner, "failed": int(failed or 0), "total": int(total or 0)})
        rows += _python_checks(con)
        for fn in extra or []:
            rows += fn(con)
    finally:
        con.close()
    df = pd.DataFrame(rows)
    df["failed_pct"] = df["failed"] / df["total"].where(df["total"] > 0)
    df["status"] = df["failed"].map(lambda n: "PASS" if n == 0 else "FAIL")
    sev = {"high": 0, "medium": 1, "low": 2}
    return (df.assign(_s=df["severity"].map(sev))
            .sort_values(["status", "_s", "failed_pct"], ascending=[True, True, False], kind="mergesort")
            .drop(columns="_s").reset_index(drop=True))


def state_cleanup_examples(db_path: Path, limit: int = 12) -> pd.DataFrame:
    """Show raw state values the cleaner fixed (raw != clean) - proof that normalisation works."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(f"""
            select state_raw, state as cleaned_to, count(*) as companies
            from stg_companies
            where state_raw is not null and (state is null or state_raw <> state)
            group by 1, 2 order by 3 desc, 1 limit {int(limit)}
        """).df()
    finally:
        con.close()

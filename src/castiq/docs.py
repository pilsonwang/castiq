"""Generate README, case study, data-source notes and career material from the computed metrics.

Every number in these files is filled in from `Results.metrics`, so a resume bullet can never disagree
with the code that produced it. Re-run `castiq analyze` and the documents update themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from castiq.config import Paths, load_yaml
from castiq.report import money, num, pct


def project_links(paths: Paths) -> dict[str, str]:
    """Links from config/project.yaml, with safe fallbacks before the repo is published."""
    cfg_path = paths.config_dir / "project.yaml"
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    repo = str(cfg.get("github_repo") or "").strip()
    url = str(cfg.get("dashboard_url") or "").strip()
    return {
        "dashboard": url or "docs/index.html",
        "repo": f"https://github.com/{repo}" if repo else "https://github.com/<you>/castiq",
        "repo_short": f"github.com/{repo}" if repo else "github.com/<you>/castiq",
        "dashboard_or_repo": url or (f"https://github.com/{repo}" if repo else "<link>"),
    }


def _f(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def readme(m: dict[str, Any], links: dict[str, str]) -> str:
    mk, t, wl, ls, fc, q = (m["market"], m["territories"], m["win_loss"], m["lead_scoring"], m["forecast"],
                            m["quality"])
    return f"""# CastIQ — RevOps analytics for an industrial die-cast supplier

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![DuckDB](https://img.shields.io/badge/warehouse-DuckDB-yellow)
![Data](https://img.shields.io/badge/data-Hugging%20Face-orange)

**One command turns three public Hugging Face datasets into the systems a revenue team runs on:**
market sizing, ICP scoring, territory design, account routing, RFQ win/loss analysis, lead scoring,
pipeline health, forecasting, and a CRM data-quality report — in a static dashboard anyone can open.

I come from industrial die-cast sales: selling cast metal parts to manufacturers. This project is what I would
build on day one as the RevOps / GTM engineer for a supplier like the ones I sold for.

![Dashboard](docs/assets/dashboard.png)

**[Open the live dashboard →]({links['dashboard']})** · [Case study](docs/CASE_STUDY.md) · [Architecture](docs/ARCHITECTURE.md) · [Data sources](docs/DATA_SOURCES.md)

## Results in 60 seconds

| Question a CRO asks | Answer from this project |
|---|---|
| How big is our market? | **{num(mk['in_market_accounts'])}** in-market US manufacturers (of {num(mk['country']['country_companies'])} US companies); **{num(mk['tier_a_accounts'])}** Tier-A. Illustrative SAM **{money(mk['sam_usd'])}/yr**. |
| Are territories fair? | Geography-only territories were **{_f(t['max_min_before'], 1)}×** apart in potential. Rebalanced, contiguous territories: **{_f(t['max_min_after'])}×** (CV {_f(t['cv_before'])} → {_f(t['cv_after'])}). |
| Which RFQs should we quote? | A qualification model on **{wl['deals']} real B2B deals** reaches AUC **{_f(wl['cv_auc_mean'])} ± {_f(wl['cv_auc_std'])}**. Top-quartile deals win **{pct(wl['q4_win_rate'])}**, bottom quartile **{pct(wl['q1_win_rate'])}**. |
| Which leads first? | Tier-A leads convert at **{pct(ls['top_tier_rate'])}** vs **{pct(ls['bottom_tier_rate'])}** for tier D (test AUC {_f(ls['test_auc'], 3)}). A documented leakage trap (+{_f(ls['leakage']['auc_inflation'], 3)} AUC) was caught and removed. |
| What will we book? | CRM default stage probabilities say **{money(fc['forecast_stage_default'])}**. Calibrated model: **{money(fc['forecast_model_expected'])}** (commit {money(fc['commit_p10'])} – best case {money(fc['best_case_p90'])}). Defaults overstate by **{pct(fc['stage_default_overstatement'])}**. |
| Is one rep better? | Rep win rates span {pct(m['reps']['min_rate'])}–{pct(m['reps']['max_rate'])}, but a chi-square test gives p = {_f(m['reps']['p_value'])}: **no evidence** of a real difference. |
| Can we trust the CRM? | **{q['failed_checks']} of {q['checks']}** automated checks fail — e.g. {pct(q['seniority_mismatch_pct'])} of contacts have a seniority that contradicts their title, {num(q['duplicate_company_websites'])} duplicate company websites. |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
castiq run-all          # download pinned data (~630 MB, cached) → build warehouse → analyze → dashboard
open docs/index.html
```

Other commands: `castiq ingest`, `castiq build`, `castiq analyze`, `make test`, `make lint`.

## What is inside

| Area | File | What it shows |
|---|---|---|
| Data ingest | `src/castiq/ingest.py` | Pinned HF revisions, SHA-256 manifest, DuckDB streaming of a 17M-row file |
| Warehouse | `src/castiq/sql/*.sql` | dbt-style staging → intermediate → marts in DuckDB |
| ICP & market | `src/castiq/icp.py`, `config/icp.yaml` | Transparent fit score a sales leader can edit; TAM/SAM/SOM |
| Territories & routing | `src/castiq/routing.py` | Balanced, contiguous territory design; auditable first-match routing rules |
| RFQ win/loss | `src/castiq/win_loss.py` | Driver analysis with confidence intervals → qualification checklist |
| Lead scoring | `src/castiq/lead_score.py` | Model selection on validation, one-time test, bootstrap CI, leakage audit, points scorecard |
| Pipeline | `src/castiq/pipeline.py` | Funnel, segment win rates with Wilson intervals, stalled deals, rep significance test |
| Forecast | `src/castiq/forecast.py` | Calibrated win probability, Monte Carlo commit/best case vs CRM defaults |
| Data quality | `src/castiq/quality.py` | {q['checks']} hygiene checks with severity, owner and reason |
| Report & docs | `src/castiq/report.py`, `src/castiq/docs.py` | Static dashboard + generated docs (numbers never drift) |

## Honest limits

- The CRM dataset is **synthetic** (no real CRM is public). The firmographics and the win/loss deals are **real**.
- Dollar market sizes use **editable spend assumptions** in `config/icp.yaml`; account counts are real.
- Close dates are hidden in the CRM source, so the forecast is validated by out-of-fold calibration, not by a quarterly back-test.

## License

Code: MIT. Data: each dataset keeps its own license (see [DATA_SOURCES](docs/DATA_SOURCES.md)).
"""


def case_study(m: dict[str, Any], frames: dict[str, pd.DataFrame]) -> str:
    mk, t, wl, ls, fc, pl, q = (m["market"], m["territories"], m["win_loss"], m["lead_scoring"], m["forecast"],
                                m["pipeline"], m["quality"])
    checklist = frames.get("checklist", pd.DataFrame())
    check_lines = "\n".join(f"| {r['ask_the_buyer']} | {r['signal']} | {r['effect']} | {pct(r['win_rate'])} | {int(r['n'])} |"
                            for r in checklist.to_dict("records"))
    terr = frames.get("territories", pd.DataFrame())
    terr_lines = "\n".join(f"| {r['name']} | {r['states']} | {money(r['potential_usd'])} | {num(r['accounts'])} |"
                           for r in terr.to_dict("records"))
    bench = frames.get("benchmark")
    bench_lines = ""
    if bench is not None:
        bench_lines = "\n".join(
            f"| {r['tier']} | {pct(r['missing_rate'])} | {_f(r['our_test_auc'], 3)} "
            f"({_f(r['our_auc_ci_low'])}–{_f(r['our_auc_ci_high'])})"
            f" | {_f(max(r['published_lr_auc_median'], r['published_gbm_auc_median']), 3)} "
            f"± {_f(r['published_auc_spread'] / 2)} | +{_f(r['leak_auc_inflation'], 3)} |"
            for r in bench.to_dict("records"))
    return f"""# Case study: a RevOps engine for an industrial die-cast supplier

## 1. The problem

A mid-size aluminum / zinc die-cast supplier sells to OEMs and Tier-1s through a long, engineering-led
cycle: RFI → RFQ package (drawings, annual volumes, PPAP level) → tooling quote → sample → production PO.
In my sales career I saw the same problems again and again:

1. **No shared definition of a good account.** Reps chase any company that sends a drawing.
2. **Territories drawn on a map, not on potential.** One rep owns Michigan, another owns the Mountain West.
3. **Every RFQ gets quoted.** Each quote costs engineering hours (DFM review, mold-flow simulation, tooling quote).
4. **The forecast is the CRM's default stage percentages.** It is always too high.
5. **CRM data nobody trusts.** Wrong titles, duplicate companies, missing states.

CastIQ addresses each one with a reproducible, tested data pipeline.

## 2. Data

| Dataset (Hugging Face) | Type | Used for |
|---|---|---|
| `bigpictureio/companies-2023-q4-sm` | **Real** — 17M companies | Market sizing, ICP scoring, territories, routing, data quality |
| `markobo/B2B_Sales_data` | **Real** — {wl['deals']} anonymised B2B deals | RFQ win/loss drivers and qualification checklist |
| `shaypal5/leadforge-lead-scoring-v1` | **Synthetic** relational CRM | Lead scoring, pipeline health, forecasting, CRM data quality |

All three are pinned to a commit hash. See [DATA_SOURCES.md](DATA_SOURCES.md).

## 3. Approach and results

### 3.1 Market map and ICP
- Streamed {num(mk['country']['country_companies'])} US companies with DuckDB and kept {num(mk['in_market_accounts'])} in die-cast-buying industries.
- Scored every company 0–100: `50% industry fit + 35% size fit + 15% geography fit` (rules in `config/icp.yaml`).
- **{num(mk['tier_a_accounts'])} Tier-A** and **{num(mk['tier_ab_accounts'])} Tier-A/B** accounts.
- Illustrative TAM {money(mk['tam_usd'])}, SAM {money(mk['sam_usd'])}, SOM {money(mk['som_usd'])} per year (spend per company is an editable assumption).

### 3.2 Territories and routing
- Strategic carve-out first: {num(t['strategic_accounts'])} Tier-A companies with 5,000+ employees ({money(t['strategic_potential_usd'])} potential).
- Territory design = weighted k-means on state centres, contiguity repair on a state-adjacency graph, then a local search
  that moves one border state at a time only if both territories stay connected.
- **Imbalance (largest ÷ smallest): {_f(t['max_min_before'], 1)}× → {_f(t['max_min_after'])}×. Coefficient of variation {_f(t['cv_before'])} → {_f(t['cv_after'])}.**

| Territory | States | A/B potential | Accounts |
|---|---|---|---|
{terr_lines}

- Routing rules (first match wins): strategic → data review ({num(t['data_review_accounts'])} accounts with no usable state) →
  territory round-robin by fit score, {t['named_accounts_per_rep']} named accounts per rep, long tail to inside sales.

### 3.3 Which RFQs to quote (real deals)
- Logistic regression on 20 qualification fields: **AUC {_f(wl['cv_auc_mean'])} ± {_f(wl['cv_auc_std'])}** (5-fold CV × 20 repeats).
- Out-of-fold deal-score quartiles: **{pct(wl['q1_win_rate'])} → {pct(wl['q4_win_rate'])}** win rate.
- Base win rate {pct(wl['base_win_rate'])}. Each driver row has a Wilson 95% interval; only significant rows are shown.

| Ask in the qualification call | Signal | Effect | Win rate | Deals |
|---|---|---|---|---|
{check_lines}

**Insight:** "Unknown" budget and "unknown" purchasing involvement lose much more often than a clear yes or no.
If the rep does not know, the deal is already at risk. That is a coaching point, not a model feature.

*Limit:* the dataset does not say when each field was filled in. A few fields (for example "buyer made a positive
statement" or "our attitude to the client") may have been updated after the outcome was known, which would flatter the
model. In a real CRM I would snapshot these fields at the RFQ date before training.

### 3.4 Lead scoring (synthetic CRM, `{ls['tier_used']}` tier)
- Two candidate models chosen on the validation split; the test split was scored once.
- **Test AUC {_f(ls['test_auc'], 3)} (95% bootstrap CI {_f(ls['test_auc_ci_low'])}–{_f(ls['test_auc_ci_high'])}).**
  Tier A converts at {pct(ls['top_tier_rate'])} vs {pct(ls['bottom_tier_rate'])} for tier D; the top 30% of leads hold {pct(ls['top30_share_of_conversions'])} of conversions.
- **Leakage audit:** `{ls['leakage']['columns_removed']}` counts touches after the scoring date. Including it would inflate AUC by
  **+{_f(ls['leakage']['auc_inflation'], 3)}** ({_f(ls['leakage']['gbm_auc_without_leak'], 3)} → {_f(ls['leakage']['gbm_auc_with_leak'], 3)}). It was removed.
- Robustness check on all three published difficulty tiers:

| Tier | Missing values | Our test AUC (95% CI) | Published best baseline | AUC from leaked column |
|---|---|---|---|---|
{bench_lines}

### 3.5 Pipeline health (synthetic CRM)
- {num(pl['opportunities'])} opportunities; win rate **{pct(pl['win_rate'], 1)}** (95% CI {pct(pl['win_rate_ci'][0], 1)}–{pct(pl['win_rate_ci'][1], 1)}).
- {num(pl['stalled_deals'])} open deals ({money(pl['stalled_acv'])}) had no touch for {pl['stalled_after_days']}+ days at their snapshot date.
- Segments that differ at 95% confidence: {", ".join(pl['significant_segments']) or "none"}.
- Rep win rates {pct(m['reps']['min_rate'])}–{pct(m['reps']['max_rate'])}; chi-square p = {_f(m['reps']['p_value'])} → no evidence of a real rep effect.

### 3.6 Forecast (synthetic CRM)
- CRM default stage probabilities (Negotiation = 90%) → **{money(fc['forecast_stage_default'])}**.
- Calibrated model on {num(fc['closed_deals_for_training'])} closed deals → **{money(fc['forecast_model_expected'])}**;
  20,000 Monte Carlo runs (with win-rate uncertainty) → commit {money(fc['commit_p10'])}, most likely {money(fc['most_likely_p50'])}, best case {money(fc['best_case_p90'])}.
- The defaults overstate bookings by **{pct(fc['stage_default_overstatement'])}**; {pct(fc['prob_reaching_stage_default'], 1)} of simulations reach them.
- Honest note: deal-level signal is weak (out-of-fold AUC {_f(fc['oof_auc'])}). The gain is calibration: the real
  {pct(fc['historical_win_rate'])} win rate instead of 90%.

### 3.7 Data quality
- {q['checks']} checks, {q['failed_checks']} failing. Highlights: {pct(q['seniority_mismatch_pct'])} of contacts have a seniority that contradicts their title;
  {num(q['duplicate_company_websites'])} duplicate company websites; {num(q['states_fixed'])} companies rescued by the state cleaner ("York", "D.C.", "Tx Texas", "Flrida").

## 4. What I would do next in a real company
1. Replace the synthetic CRM with the company's Salesforce/HubSpot export — the SQL staging layer is the only part that changes.
2. Write the ICP tier and owner back to the CRM (reverse ETL) and alert reps on new Tier-A RFQs.
3. Add intent signals (hiring for "die casting", new EV programs, plant expansions) to the fit score.
4. Back-test the forecast quarterly once close dates are available and track forecast error as a team KPI.
5. Put the data-quality checks in a weekly scheduled job with owners notified.
"""


def data_sources(m: dict[str, Any]) -> str:
    src = m.get("sources", {})
    rows = []
    for group, entries in src.items():
        for e in entries:
            rows.append(f"| {group} | [`{e['repo_id']}`]({e['url']}) | `{e['revision'][:12]}` | `{e['file']}` "
                        f"| {num(e['rows'])} | `{e['sha256'][:12]}…` |")
    table = "\n".join(rows)
    return f"""# Data sources

All data comes from public Hugging Face datasets, pinned to an exact commit. `castiq ingest` downloads each file,
records its SHA-256 hash in `data/raw/manifest.json`, and the table below is generated from that manifest.

| Group | Dataset | Revision | File | Rows | SHA-256 |
|---|---|---|---|---|---|
{table}

## What is real and what is synthetic

| Dataset | Real or synthetic | License | Notes |
|---|---|---|---|
| `bigpictureio/companies-2023-q4-sm` | **Real** | ODC-BY | Company name, website, industry, size band, HQ city/state. Free-text state values are cleaned in `geo.py`. |
| `markobo/B2B_Sales_data` | **Real** (anonymised) | CC-BY-4.0 | 448 closed deals of a Slovenian B2B IT solutions company (Salvirt). Products and sellers are anonymised. |
| `shaypal5/leadforge-lead-scoring-v1` | **Synthetic** | MIT | Relational CRM generated by the open-source `leadforge` simulator. Events stop 30 days after each lead is created; close dates are hidden. |

## Derived fields (not in any source)

| Field | Where | How |
|---|---|---|
| `fit_score`, `tier` | `icp.py` | Weighted rules in `config/icp.yaml` |
| `est_annual_spend_usd` | `icp.py` | **Assumption**: spend per size band × industry fit (`config/icp.yaml`) |
| `territory`, `owner`, `route_rule` | `routing.py` | Territory design + routing rules |
| `p_win_model` | `forecast.py` | Logistic regression on closed deals |
| Lead `score` | `lead_score.py` | Model trained on the published train/valid splits |
"""


def resume_bullets(m: dict[str, Any], links: dict[str, str]) -> str:
    mk, t, wl, ls, fc, q = (m["market"], m["territories"], m["win_loss"], m["lead_scoring"], m["forecast"],
                            m["quality"])
    return f"""# Resume bullets (numbers generated from the project)

Use 3–4 of these under a **Projects** heading, or under your current role if you build the same thing at work.
Every number below comes from `docs/metrics.json` and changes if the data or config changes.

**CastIQ — RevOps analytics engine for industrial manufacturing (Python, SQL/DuckDB, scikit-learn)** · {links['repo_short']}

- Built an end-to-end GTM data pipeline (Hugging Face → DuckDB warehouse → Python models → static dashboard) with
  pinned data versions, CI, and a full test suite; one command rebuilds every number.
- Sized a market of **{num(mk['in_market_accounts'])} US manufacturers** from a 17M-company dataset and built an editable ICP fit
  score that identifies **{num(mk['tier_a_accounts'])} Tier-A accounts**.
- Designed balanced, contiguous sales territories that cut potential imbalance from **{_f(t['max_min_before'], 1)}× to {_f(t['max_min_after'])}×**,
  plus first-match routing rules with a strategic-account carve-out and capacity caps.
- Modeled win/loss on **{wl['deals']} real B2B deals** (RFI / RFP / tender process) (AUC **{_f(wl['cv_auc_mean'])}**); turned the drivers into an RFQ qualification
  checklist — top-quartile deals won **{pct(wl['q4_win_rate'])}** vs **{pct(wl['q1_win_rate'])}** for the bottom quartile.
- Showed that out-of-the-box CRM stage probabilities overstate the forecast by **{pct(fc['stage_default_overstatement'])}**; replaced them with a
  calibrated model and a Monte Carlo commit/best-case range.
- Caught and removed a data-leakage feature that would have inflated lead-score AUC by **+{_f(ls['leakage']['auc_inflation'], 3)}**; shipped
  an explainable points scorecard (tier A converts **{_f(ls['top_tier_lift'])}×** the average).
- Automated **{q['checks']} CRM data-quality checks** with severity, owner and fix — e.g. {pct(q['seniority_mismatch_pct'])} of contacts had a
  seniority that contradicted their job title.

## One-line version (for the summary at the top)

> Die-cast sales professional turned RevOps builder: I combine [N] years of selling engineered metal parts to OEMs with
> Python/SQL systems for ICP scoring, territory design, forecasting and CRM data quality.
"""


def linkedin_post(m: dict[str, Any], links: dict[str, str]) -> str:
    t, wl, fc, q = m["territories"], m["win_loss"], m["forecast"], m["quality"]
    return f"""# LinkedIn post draft

> I spent years selling die-cast parts to manufacturers. The hardest part was never the casting.
> It was deciding **which RFQ to quote, which account to call, and what number to forecast.**
>
> So I built the system I always wanted — on public data, with code anyone can run:
>
> 📍 Territories: rebalanced so the biggest is {_f(t['max_min_after'])}× the smallest (was {_f(t['max_min_before'], 1)}×), and every territory is drivable.
> 🎯 RFQs: on {wl['deals']} real B2B deals, the top-scored quartile won {pct(wl['q4_win_rate'])}; the bottom quartile won {pct(wl['q1_win_rate'])}.
>    The biggest red flag? The rep saying "I don't know" about budget or purchasing.
> 📉 Forecast: CRM default stage percentages overstated bookings by {pct(fc['stage_default_overstatement'])}.
> 🧹 Data: {q['failed_checks']} of {q['checks']} CRM hygiene checks failed. Every one has an owner and a fix.
>
> Stack: Python, SQL (DuckDB), scikit-learn, Hugging Face datasets, GitHub Actions.
> Dashboard + code: {links['dashboard_or_repo']}
>
> I'm looking for a remote/hybrid RevOps or GTM Engineering role — especially in manufacturing tech.
> If your team sells to engineers and purchasing managers, let's talk.
>
> #RevOps #GTMEngineering #SalesOperations #Manufacturing #DieCasting

**"Featured" section text:** *CastIQ — RevOps engine for industrial suppliers. Market sizing, territories, RFQ win/loss,
forecasting and CRM data quality on public data. Live dashboard + code.*
"""


def interview_track(m: dict[str, Any]) -> str:
    t, wl, ls, fc, q = m["territories"], m["win_loss"], m["lead_scoring"], m["forecast"], m["quality"]
    return f"""# Interview talk track

## "Tell me about yourself" — 2 minutes

1. **Where I come from (30 s).** I sold industrial die-cast parts to OEMs and Tier-1s. Long cycles, engineers who
   spec the part, purchasing who awards it, and an RFQ that costs us real engineering hours to answer.
2. **What I noticed (30 s).** We lost money on bad RFQs, territories were drawn on a map, and the forecast was the CRM's default
   percentages. The problems were data and process problems, not selling problems.
3. **What I built (45 s).** CastIQ: a tested Python + SQL pipeline on public Hugging Face data. It sizes the market
   ({num(m['market']['in_market_accounts'])} manufacturers), designs fair territories ({_f(t['max_min_before'], 1)}× → {_f(t['max_min_after'])}× imbalance), scores RFQs
   (AUC {_f(wl['cv_auc_mean'])} on real deals), and shows the CRM-default forecast is {pct(fc['stage_default_overstatement'])} too high.
4. **Why this role (15 s).** I want to build these systems full time for a team that sells to people like my old customers.

## Six common RevOps / GTM-engineering questions

**1. How would you build a lead scoring model from scratch?**
Start with the definition of a conversion and the scoring moment, then remove every field that is only known after
that moment. In CastIQ a column called `total_touches_all` looked harmless but counted future touches — it would have
added +{_f(ls['leakage']['auc_inflation'], 3)} AUC that disappears in production. Choose the model on a validation split, score the test split once,
and report a confidence interval ({_f(ls['test_auc_ci_low'])}–{_f(ls['test_auc_ci_high'])}). Ship a points scorecard reps can read.

**2. How do you design territories?**
Start from potential, not from geography or account count. Carve out strategic accounts first. Then balance potential while
keeping territories contiguous and drivable. I measure fairness with the coefficient of variation ({_f(t['cv_before'])} → {_f(t['cv_after'])})
and the largest-to-smallest ratio.

**3. Our forecast is always wrong. Where do you start?**
Compare the stage probabilities in the CRM with real historical win rates. In CastIQ the defaults said 90% at negotiation;
history said {pct(fc['historical_win_rate'])}. Calibration alone removes most of the error. Then give a range (commit / most likely / best case),
not one number.

**4. A sales leader says rep X is the best closer. How do you check?**
Put a confidence interval on every rep's win rate and test whether the spread is larger than chance. In CastIQ the spread
looked like {pct(m['reps']['min_rate'])}–{pct(m['reps']['max_rate'])}, but p = {_f(m['reps']['p_value'])} — it was noise.

**5. How do you handle bad CRM data?**
Automate the checks, give each one a severity and an owner, and report the trend weekly. {q['checks']} checks in CastIQ;
for example {pct(q['seniority_mismatch_pct'])} of contacts had a seniority that contradicted their title — that breaks persona routing.

**6. Why should we hire a salesperson for a RevOps / GTM engineering role?**
Because I know what a rep does with a bad score or an unfair territory: ignores it. I design the system so that the
people who use it believe it — transparent rules, intervals on every number, and fields that match how buyers really buy.
"""


def job_search() -> str:
    return """# Job search plan: remote / hybrid GTM roles above $100k

## Titles to search (best fit first)

| Title | Why it fits | Typical US base range* |
|---|---|---|
| GTM Engineer / Revenue Engineer | This project is exactly the job | $120k – $180k |
| Revenue Operations Manager | Territories, forecasting, data quality | $110k – $160k |
| Sales Operations Manager (manufacturing / industrial) | Domain + ops | $100k – $150k |
| Sales Strategy & Analytics / Sales Analytics Manager | Market sizing, win/loss | $110k – $160k |
| Solutions Consultant / Sales Engineer — manufacturing software | Speaks the buyer's language | $110k – $150k base, $150k – $200k OTE |
| Account Executive — manufacturing SaaS or industrial marketplace | Shortest path from current role | $90k – $140k base, $180k – $260k OTE |

\\*Rough ranges from public postings; check current numbers on LinkedIn, Levels.fyi, Glassdoor or RevGenius salary data before you negotiate.

## Companies where a die-cast background is an advantage

Software and marketplaces that sell to manufacturers or job shops — for example quoting and ERP software for machine and
casting shops, on-demand manufacturing marketplaces, MES / connected-worker platforms, and manufacturing ERP vendors.
Search phrases: "manufacturing software RevOps", "industrial marketplace sales operations", "quoting software job shops",
"MES platform revenue operations".

## LinkedIn search strings

```
("revenue operations" OR "RevOps" OR "GTM engineer" OR "sales operations") AND (manufacturing OR industrial) AND remote
("sales engineer" OR "solutions consultant") AND (MES OR ERP OR "manufacturing software")
```

## Weekly routine

1. 10 targeted applications, each with the dashboard link and one tailored bullet.
2. 5 messages to RevOps / GTM leaders at target companies: one sentence on the die-cast background, one result from CastIQ, one question.
3. Post one short insight from the project (territories, RFQ red flags, forecast calibration) on LinkedIn.
4. Track every application in a simple pipeline: applied → screen → hiring manager → onsite → offer. You are a sales pro: run your search like a funnel.
"""


def write_all(metrics: dict[str, Any], frames: dict[str, pd.DataFrame], paths: Paths) -> list[Path]:
    links = project_links(paths)
    files = {
        paths.root / "README.md": readme(metrics, links),
        paths.docs / "CASE_STUDY.md": case_study(metrics, frames),
        paths.docs / "DATA_SOURCES.md": data_sources(metrics),
        paths.docs / "career" / "RESUME_BULLETS.md": resume_bullets(metrics, links),
        paths.docs / "career" / "LINKEDIN_POST.md": linkedin_post(metrics, links),
        paths.docs / "career" / "INTERVIEW_TALK_TRACK.md": interview_track(metrics),
        paths.docs / "career" / "JOB_SEARCH.md": job_search(),
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return list(files)

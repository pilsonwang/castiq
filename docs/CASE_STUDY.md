# Case study: a RevOps engine for an industrial die-cast supplier

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
| `markobo/B2B_Sales_data` | **Real** — 448 anonymised B2B deals | RFQ win/loss drivers and qualification checklist |
| `shaypal5/leadforge-lead-scoring-v1` | **Synthetic** relational CRM | Lead scoring, pipeline health, forecasting, CRM data quality |

All three are pinned to a commit hash. See [DATA_SOURCES.md](DATA_SOURCES.md).

## 3. Approach and results

### 3.1 Market map and ICP
- Streamed 4,306,855 US companies with DuckDB and kept 247,552 in die-cast-buying industries.
- Scored every company 0–100: `50% industry fit + 35% size fit + 15% geography fit` (rules in `config/icp.yaml`).
- **20,309 Tier-A** and **90,822 Tier-A/B** accounts.
- Illustrative TAM $21.7B, SAM $7.6B, SOM $226.7M per year (spend per company is an editable assumption).

### 3.2 Territories and routing
- Strategic carve-out first: 898 Tier-A companies with 5,000+ employees ($6.8B potential).
- Territory design = weighted k-means on state centres, contiguity repair on a state-adjacency graph, then a local search
  that moves one border state at a time only if both territories stay connected.
- **Imbalance (largest ÷ smallest): 9.8× → 1.34×. Coefficient of variation 0.51 → 0.10.**

| Territory | States | A/B potential | Accounts |
|---|---|---|---|
| T4 · NY/PA/MA | CT, DC, DE, MA, MD, ME, NH, NJ, NY, PA, RI, VT | $2.6B | 14,559 |
| T6 · FL/GA/VA | AL, FL, GA, KY, NC, SC, VA, WV | $2.4B | 15,624 |
| T1 · MI/OH/WI | MI, MN, ND, OH, SD, WI | $2.4B | 17,275 |
| T5 · IL/IN/MO | AR, IA, IL, IN, KS, LA, MO, MS, NE, OK, TN | $2.2B | 16,052 |
| T3 · CA/HI | CA, HI | $2.0B | 10,644 |
| T2 · TX/WA/AZ | AK, AZ, CO, ID, MT, NM, NV, OR, TX, UT, WA, WY | $2.0B | 15,004 |

- Routing rules (first match wins): strategic → data review (766 accounts with no usable state) →
  territory round-robin by fit score, 150 named accounts per rep, long tail to inside sales.

### 3.3 Which RFQs to quote (real deals)
- Logistic regression on 20 qualification fields: **AUC 0.81 ± 0.04** (5-fold CV × 20 repeats).
- Out-of-fold deal-score quartiles: **13% → 85%** win rate.
- Base win rate 51%. Each driver row has a Wilson 95% interval; only significant rows are shown.

| Ask in the qualification call | Signal | Effect | Win rate | Deals |
|---|---|---|---|---|
| Existing relationship / referral / trade show | Where the deal came from = Direct mail | lowers win rate by 51 pts | 0% | 16 |
| Is the program funded (tooling PO budget approved)? | Budget allocated = Unknown | lowers win rate by 37 pts | 14% | 36 |
| New program / re-source / maintenance | Deal type = Maintenance | raises win rate by 36 pts | 87% | 31 |
| Strategic account vs. normal vs. first deal | Our attitude to the client = First deal | lowers win rate by 35 pts | 15% | 39 |
| Is Purchasing running the sourcing event? | Purchasing department involved = Unknown | lowers win rate by 35 pts | 16% | 32 |
| Already an approved supplier at this plant? | New or current client = New | lowers win rate by 34 pts | 17% | 158 |
| Are other casters quoting? | Competitors in the deal = Yes | lowers win rate by 30 pts | 21% | 143 |
| Is this a strategic program for us? | Strategic importance to us = Unimportant | lowers win rate by 29 pts | 21% | 28 |

**Insight:** "Unknown" budget and "unknown" purchasing involvement lose much more often than a clear yes or no.
If the rep does not know, the deal is already at risk. That is a coaching point, not a model feature.

*Limit:* the dataset does not say when each field was filled in. A few fields (for example "buyer made a positive
statement" or "our attitude to the client") may have been updated after the outcome was known, which would flatter the
model. In a real CRM I would snapshot these fields at the RFQ date before training.

### 3.4 Lead scoring (synthetic CRM, `intermediate` tier)
- Two candidate models chosen on the validation split; the test split was scored once.
- **Test AUC 0.606 (95% bootstrap CI 0.56–0.65).**
  Tier A converts at 33% vs 18% for tier D; the top 30% of leads hold 39% of conversions.
- **Leakage audit:** `total_touches_all` counts touches after the scoring date. Including it would inflate AUC by
  **+0.035** (0.610 → 0.645). It was removed.
- Robustness check on all three published difficulty tiers:

| Tier | Missing values | Our test AUC (95% CI) | Published best baseline | AUC from leaked column |
|---|---|---|---|---|
| intro | 2% | 0.640 (0.60–0.68) | 0.684 ± 0.06 | +0.031 |
| intermediate | 8% | 0.606 (0.56–0.65) | 0.662 ± 0.03 | +0.035 |
| advanced | 18% | 0.534 (0.47–0.59) | 0.624 ± 0.05 | +0.030 |

### 3.5 Pipeline health (synthetic CRM)
- 4,255 opportunities; win rate **48.0%** (95% CI 45.9%–50.2%).
- 132 open deals ($7.8M) had no touch for 7+ days at their snapshot date.
- Segments that differ at 95% confidence: Lead source: partner_referral, Industry: manufacturing, Buyer role of first contact: end_user.
- Rep win rates 45%–50%; chi-square p = 1.00 → no evidence of a real rep effect.

### 3.6 Forecast (synthetic CRM)
- CRM default stage probabilities (Negotiation = 90%) → **$117.6M**.
- Calibrated model on 2,099 closed deals → **$65.0M**;
  20,000 Monte Carlo runs (with win-rate uncertainty) → commit $62.3M, most likely $65.0M, best case $67.7M.
- The defaults overstate bookings by **81%**; 0.0% of simulations reach them.
- Honest note: deal-level signal is weak (out-of-fold AUC 0.57). The gain is calibration: the real
  48% win rate instead of 90%.

### 3.7 Data quality
- 14 checks, 8 failing. Highlights: 78% of contacts have a seniority that contradicts their title;
  10,755 duplicate company websites; 527 companies rescued by the state cleaner ("York", "D.C.", "Tx Texas", "Flrida").

## 4. What I would do next in a real company
1. Replace the synthetic CRM with the company's Salesforce/HubSpot export — the SQL staging layer is the only part that changes.
2. Write the ICP tier and owner back to the CRM (reverse ETL) and alert reps on new Tier-A RFQs.
3. Add intent signals (hiring for "die casting", new EV programs, plant expansions) to the fit score.
4. Back-test the forecast quarterly once close dates are available and track forecast error as a team KPI.
5. Put the data-quality checks in a weekly scheduled job with owners notified.

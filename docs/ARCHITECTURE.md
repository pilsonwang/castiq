# Architecture

```mermaid
flowchart LR
    subgraph HF["Hugging Face (pinned commits)"]
        A["bigpictureio/companies-2023-q4-sm<br/>17M real companies"]
        B["markobo/B2B_Sales_data<br/>448 real B2B deals"]
        C["shaypal5/leadforge-lead-scoring-v1<br/>synthetic relational CRM"]
    end
    subgraph ING["ingest.py"]
        D["download + SHA-256 manifest"]
        E["DuckDB stream filter<br/>US + in-market industries<br/>state cleaning (geo.py)"]
    end
    subgraph WH["DuckDB warehouse (sql/*.sql)"]
        F["01 staging"] --> G["02 intermediate<br/>engagement, committee,<br/>per-lead snapshot"] --> H["03 marts<br/>opportunities, funnel, reps"]
    end
    subgraph AN["analyze.py"]
        I["icp.py<br/>fit score, TAM/SAM/SOM"]
        J["routing.py<br/>territories + routing"]
        K["win_loss.py<br/>RFQ drivers + checklist"]
        L["lead_score.py + benchmark.py<br/>model, leakage audit"]
        M["pipeline.py<br/>funnel, win rates, reps"]
        N["forecast.py<br/>calibration + Monte Carlo"]
        O["quality.py<br/>hygiene checks"]
    end
    P["docs/metrics.json<br/>(single source of truth)"]
    Q["report.py → docs/index.html"]
    R["docs.py → README, case study,<br/>data sources, career docs"]

    A --> E --> F
    B --> D --> F
    C --> D
    H --> I & J & K & M & N & O
    D --> L
    I --> J
    AN --> P --> Q & R
```

## Design decisions

| Decision | Why |
|---|---|
| **DuckDB** as the warehouse | Streams a 629 MB gzip CSV in ~20 s on a laptop; the SQL ports to Snowflake / BigQuery with small changes. |
| **dbt-style SQL layers** without dbt | Staging → intermediate → marts is the pattern a data team expects; a 40-line runner avoids a heavy dependency. |
| **Pinned Hugging Face revisions + SHA-256 manifest** | Every number can be traced to a byte-identical input. |
| **Config in YAML** (`config/*.yaml`) | Sales leaders edit ICP weights, stage probabilities and territory count without touching code. |
| **One metrics file feeds every output** | The dashboard, README, case study and resume bullets can never disagree. |
| **Intervals on every rate** | Small samples lie. Wilson intervals, bootstrap AUC intervals and a chi-square test stop false conclusions. |
| **Static HTML dashboard** | Opens anywhere (GitHub Pages, email attachment, offline). No server, no login. |
| **Leakage audit inside the model code** | The leaked-feature model is trained on purpose so the inflation is measured, not guessed. |

## Data model (marts)

| Table | Grain | Key columns |
|---|---|---|
| `stg_companies` | company | industry, size_band, state (cleaned), state_raw |
| `mart_opportunities` | opportunity | stage, acv, is_won, account firmographics, contact persona, engagement counts, buyer roles engaged, days_since_engaged |
| `mart_funnel` | funnel step | step, n |
| `mart_rep_scorecard` | rep | opportunities, won, won_acv, open_pipeline_acv |
| `stg_deals` | real B2B deal | qualification fields, is_won |

## Tests

`tests/` holds unit tests for each module (small hand-made fixtures, no network) and one end-to-end test that runs
ingest-free warehouse build → analysis → dashboard → docs on a synthetic fixture project. Set `CASTIQ_NETWORK_TESTS=1`
to also run the test that downloads from Hugging Face and checks the pinned file hashes.

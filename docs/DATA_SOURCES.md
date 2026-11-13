# Data sources

All data comes from public Hugging Face datasets, pinned to an exact commit. `castiq ingest` downloads each file,
records its SHA-256 hash in `data/raw/manifest.json`, and the table below is generated from that manifest.

| Group | Dataset | Revision | File | Rows | SHA-256 |
|---|---|---|---|---|---|
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tables/accounts.parquet) | `0bf5ea25697d` | `intermediate/tables/accounts.parquet` | 1,500 | `79d8ed63bac1…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tables/contacts.parquet) | `0bf5ea25697d` | `intermediate/tables/contacts.parquet` | 4,200 | `51657ae49a12…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tables/leads.parquet) | `0bf5ea25697d` | `intermediate/tables/leads.parquet` | 5,000 | `d87d2e01146d…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tables/opportunities.parquet) | `0bf5ea25697d` | `intermediate/tables/opportunities.parquet` | 4,255 | `8358db176ed4…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tables/sales_activities.parquet) | `0bf5ea25697d` | `intermediate/tables/sales_activities.parquet` | 20,679 | `b5bd72c53e1d…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tables/sessions.parquet) | `0bf5ea25697d` | `intermediate/tables/sessions.parquet` | 10,012 | `aa3efc024606…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tables/touches.parquet) | `0bf5ea25697d` | `intermediate/tables/touches.parquet` | 38,724 | `d463a53386df…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tasks/converted_within_90_days/train.parquet) | `0bf5ea25697d` | `intermediate/tasks/converted_within_90_days/train.parquet` | 3,500 | `b4c8dc5d45e7…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tasks/converted_within_90_days/valid.parquet) | `0bf5ea25697d` | `intermediate/tasks/converted_within_90_days/valid.parquet` | 750 | `6845bc851c22…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tasks/converted_within_90_days/test.parquet) | `0bf5ea25697d` | `intermediate/tasks/converted_within_90_days/test.parquet` | 750 | `55cf4611f72f…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intro/tasks/converted_within_90_days/train.parquet) | `0bf5ea25697d` | `intro/tasks/converted_within_90_days/train.parquet` | 3,500 | `3a8249477166…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intro/tasks/converted_within_90_days/valid.parquet) | `0bf5ea25697d` | `intro/tasks/converted_within_90_days/valid.parquet` | 750 | `5e0c150bda66…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intro/tasks/converted_within_90_days/test.parquet) | `0bf5ea25697d` | `intro/tasks/converted_within_90_days/test.parquet` | 750 | `9df4ee41fbe2…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tasks/converted_within_90_days/train.parquet) | `0bf5ea25697d` | `intermediate/tasks/converted_within_90_days/train.parquet` | 3,500 | `b4c8dc5d45e7…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tasks/converted_within_90_days/valid.parquet) | `0bf5ea25697d` | `intermediate/tasks/converted_within_90_days/valid.parquet` | 750 | `6845bc851c22…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/intermediate/tasks/converted_within_90_days/test.parquet) | `0bf5ea25697d` | `intermediate/tasks/converted_within_90_days/test.parquet` | 750 | `55cf4611f72f…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/advanced/tasks/converted_within_90_days/train.parquet) | `0bf5ea25697d` | `advanced/tasks/converted_within_90_days/train.parquet` | 3,500 | `83e84e49ba6d…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/advanced/tasks/converted_within_90_days/valid.parquet) | `0bf5ea25697d` | `advanced/tasks/converted_within_90_days/valid.parquet` | 750 | `18a310c9dac7…` |
| crm | [`shaypal5/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/shaypal5/leadforge-lead-scoring-v1/blob/0bf5ea25697de87e37e39898e4281fd90d98ac9a/advanced/tasks/converted_within_90_days/test.parquet) | `0bf5ea25697d` | `advanced/tasks/converted_within_90_days/test.parquet` | 750 | `431d38734b13…` |
| deals | [`markobo/B2B_Sales_data`](https://huggingface.co/datasets/markobo/B2B_Sales_data/blob/c3bae20f010d8ec74f6e9f85ef19368811dd15d3/Salvirt_B2B_ML_dataset_HF.csv) | `c3bae20f010d` | `Salvirt_B2B_ML_dataset_HF.csv` | 448 | `40496dc96175…` |
| companies | [`bigpictureio/companies-2023-q4-sm`](https://huggingface.co/datasets/bigpictureio/companies-2023-q4-sm/blob/19f3f14210e05d9e2be1b6e00fbb50c9317c1414/companies-2023-q4-sm.csv.gz) | `19f3f14210e0` | `companies-2023-q4-sm.csv.gz` | 4,306,855 | `c673f1c6936c…` |

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

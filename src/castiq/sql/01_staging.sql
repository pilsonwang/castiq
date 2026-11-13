-- STAGING: one model per source table. Rename, cast types, add simple flags. No business logic.
-- {raw} is replaced with the data/raw folder at run time.

create or replace table stg_accounts as
select
    account_id,
    industry,
    region,
    employee_band,
    estimated_revenue_band,
    process_maturity_band,
    try_cast(created_at as date) as account_created_at
from read_parquet('{raw}/crm/accounts.parquet');

create or replace table stg_contacts as
select
    contact_id,
    account_id,
    job_title,
    role_function,
    seniority,
    buyer_role,
    email_domain_type,
    try_cast(created_at as date) as contact_created_at
from read_parquet('{raw}/crm/contacts.parquet');

create or replace table stg_leads as
select
    lead_id,
    contact_id,
    account_id,
    try_cast(lead_created_at as date) as lead_created_at,
    lead_source,
    first_touch_channel,
    owner_rep_id
from read_parquet('{raw}/crm/leads.parquet');

create or replace table stg_opportunities as
select
    opportunity_id,
    lead_id,
    try_cast(created_at as date) as opp_created_at,
    stage,
    cast(estimated_acv as double) as acv,
    stage in ('closed_won', 'closed_lost') as is_closed,
    stage = 'closed_won' as is_won
from read_parquet('{raw}/crm/opportunities.parquet');

create or replace table stg_activities as
select
    activity_id,
    lead_id,
    rep_id,
    try_cast(activity_timestamp as date) as activity_date,
    activity_type,
    activity_outcome
from read_parquet('{raw}/crm/sales_activities.parquet');

create or replace table stg_sessions as
select
    session_id,
    lead_id,
    try_cast(session_timestamp as date) as session_date,
    session_type,
    page_views,
    pricing_page_views,
    demo_page_views,
    session_duration_seconds
from read_parquet('{raw}/crm/sessions.parquet');

create or replace table stg_touches as
select
    touch_id,
    lead_id,
    try_cast(touch_timestamp as date) as touch_date,
    touch_type,
    touch_channel,
    touch_direction,
    campaign_id
from read_parquet('{raw}/crm/touches.parquet');

-- Real B2B deals (Salvirt). Status is the label: Won / Lost.
create or replace table stg_deals as
select
    row_number() over () as deal_id,
    *,
    Status = 'Won' as is_won
from read_parquet('{raw}/deals/deals.parquet');

-- Real US firmographics, already filtered to in-market industries during ingest.
create or replace table stg_companies as
select
    handle,
    name,
    website,
    industry,
    coalesce(size, 'unknown') as size_band,
    type as company_type,
    founded,
    city,
    state,
    state_raw
from read_parquet('{raw}/companies/companies.parquet');

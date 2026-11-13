-- MARTS: analysis-ready tables. These are what a BI tool or a CRM sync would read.

create or replace table mart_opportunities as
select
    o.opportunity_id,
    o.lead_id,
    l.account_id,
    l.contact_id,
    l.owner_rep_id,
    l.lead_source,
    l.lead_created_at,
    o.opp_created_at,
    o.stage,
    o.acv,
    o.is_closed,
    o.is_won,
    a.industry,
    a.region,
    a.employee_band,
    a.estimated_revenue_band,
    a.process_maturity_band,
    c.role_function,
    c.seniority,
    c.buyer_role,
    c.email_domain_type,
    e.touches,
    e.inbound_touches,
    e.outbound_touches,
    e.sessions,
    e.pricing_page_views,
    e.demo_page_views,
    e.session_seconds,
    e.activities,
    e.meetings,
    e.demos_completed,
    e.live_conversations,
    e.reps_involved,
    e.last_engaged_date,
    coalesce(ac.buyer_roles_engaged, 0) as buyer_roles_engaged,
    coalesce(ac.contacts_engaged, 0) as contacts_engaged,
    datediff('day', o.opp_created_at, ao.as_of_date) as days_open,
    datediff('day', coalesce(e.last_engaged_date, o.opp_created_at), ao.as_of_date) as days_since_engaged,
    datediff('day', l.lead_created_at, o.opp_created_at) as days_lead_to_opp,
    ao.as_of_date
from stg_opportunities o
join stg_leads l using (lead_id)
left join stg_accounts a on a.account_id = l.account_id
left join stg_contacts c on c.contact_id = l.contact_id
left join int_lead_engagement e on e.lead_id = o.lead_id
left join int_account_committee ac on ac.account_id = l.account_id
join int_lead_snapshot ao on ao.lead_id = o.lead_id;

create or replace table mart_funnel as
with leads as (select count(*) as n from stg_leads),
     engaged as (select count(*) as n from int_lead_engagement where live_conversations > 0),
     opps as (select count(*) as n from stg_opportunities),
     closed as (select count(*) as n from stg_opportunities where is_closed),
     won as (select count(*) as n from stg_opportunities where is_won)
select 1 as step_order, 'Leads' as step, n from leads
union all select 2, 'Live conversation', n from engaged
union all select 3, 'Opportunity', n from opps
union all select 4, 'Closed (won or lost)', n from closed
union all select 5, 'Closed won', n from won
order by step_order;

create or replace table mart_rep_scorecard as
select
    owner_rep_id as rep_id,
    count(*) as opportunities,
    count(*) filter (where is_closed) as closed,
    count(*) filter (where is_won) as won,
    sum(acv) filter (where is_won) as won_acv,
    sum(acv) filter (where not is_closed) as open_pipeline_acv,
    avg(activities) as avg_activities_per_opp
from mart_opportunities
group by 1
order by 1;

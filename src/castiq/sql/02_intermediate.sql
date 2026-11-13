-- INTERMEDIATE: reusable building blocks. One row per lead / per account.

-- "As of" date. The source exports each lead's events up to `snapshot_day` days after the lead was
-- created (a per-lead snapshot). Measuring "days since last touch" against one global date would make
-- early leads look stalled only because their export window closed sooner. So we measure each lead
-- at its own snapshot date.
create or replace table int_lead_snapshot as
select
    lead_id,
    lead_created_at + to_days({snapshot_day}) as as_of_date
from stg_leads;

create or replace table int_lead_engagement as
with t as (
    select
        lead_id,
        count(*) as touches,
        count(*) filter (where touch_direction = 'inbound') as inbound_touches,
        count(*) filter (where touch_direction = 'outbound') as outbound_touches,
        min(touch_date) as first_touch_date,
        max(touch_date) as last_touch_date
    from stg_touches
    group by 1
),
s as (
    select
        lead_id,
        count(*) as sessions,
        sum(page_views) as page_views,
        sum(pricing_page_views) as pricing_page_views,
        sum(demo_page_views) as demo_page_views,
        sum(session_duration_seconds) as session_seconds
    from stg_sessions
    group by 1
),
a as (
    select
        lead_id,
        count(*) as activities,
        count(*) filter (where activity_type = 'meeting') as meetings,
        count(*) filter (where activity_outcome = 'demo_completed') as demos_completed,
        count(*) filter (where activity_outcome in ('connected', 'meeting_set', 'demo_completed')) as live_conversations,
        count(distinct rep_id) as reps_involved,
        max(activity_date) as last_activity_date
    from stg_activities
    group by 1
)
select
    l.lead_id,
    coalesce(t.touches, 0) as touches,
    coalesce(t.inbound_touches, 0) as inbound_touches,
    coalesce(t.outbound_touches, 0) as outbound_touches,
    coalesce(s.sessions, 0) as sessions,
    coalesce(s.page_views, 0) as page_views,
    coalesce(s.pricing_page_views, 0) as pricing_page_views,
    coalesce(s.demo_page_views, 0) as demo_page_views,
    coalesce(s.session_seconds, 0) as session_seconds,
    coalesce(a.activities, 0) as activities,
    coalesce(a.meetings, 0) as meetings,
    coalesce(a.demos_completed, 0) as demos_completed,
    coalesce(a.live_conversations, 0) as live_conversations,
    coalesce(a.reps_involved, 0) as reps_involved,
    t.first_touch_date,
    greatest(t.last_touch_date, a.last_activity_date) as last_engaged_date
from stg_leads l
left join t using (lead_id)
left join s using (lead_id)
left join a using (lead_id);

-- Buying-committee coverage: how many distinct buyer roles we have a lead for at each account.
-- In industrial deals the engineer specs the part and purchasing awards it - single-threaded deals stall.
create or replace table int_account_committee as
select
    l.account_id,
    count(distinct l.contact_id) as contacts_engaged,
    count(distinct c.buyer_role) as buyer_roles_engaged,
    bool_or(c.buyer_role = 'economic_buyer') as has_economic_buyer,
    bool_or(c.buyer_role = 'champion') as has_champion
from stg_leads l
join stg_contacts c using (contact_id)
group by 1;

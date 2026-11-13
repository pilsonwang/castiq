# Interview talk track

## "Tell me about yourself" — 2 minutes

1. **Where I come from (30 s).** I sold industrial die-cast parts to OEMs and Tier-1s. Long cycles, engineers who
   spec the part, purchasing who awards it, and an RFQ that costs us real engineering hours to answer.
2. **What I noticed (30 s).** We lost money on bad RFQs, territories were drawn on a map, and the forecast was the CRM's default
   percentages. The problems were data and process problems, not selling problems.
3. **What I built (45 s).** CastIQ: a tested Python + SQL pipeline on public Hugging Face data. It sizes the market
   (247,552 manufacturers), designs fair territories (9.8× → 1.34× imbalance), scores RFQs
   (AUC 0.81 on real deals), and shows the CRM-default forecast is 81% too high.
4. **Why this role (15 s).** I want to build these systems full time for a team that sells to people like my old customers.

## Six common RevOps / GTM-engineering questions

**1. How would you build a lead scoring model from scratch?**
Start with the definition of a conversion and the scoring moment, then remove every field that is only known after
that moment. In CastIQ a column called `total_touches_all` looked harmless but counted future touches — it would have
added +0.035 AUC that disappears in production. Choose the model on a validation split, score the test split once,
and report a confidence interval (0.56–0.65). Ship a points scorecard reps can read.

**2. How do you design territories?**
Start from potential, not from geography or account count. Carve out strategic accounts first. Then balance potential while
keeping territories contiguous and drivable. I measure fairness with the coefficient of variation (0.51 → 0.10)
and the largest-to-smallest ratio.

**3. Our forecast is always wrong. Where do you start?**
Compare the stage probabilities in the CRM with real historical win rates. In CastIQ the defaults said 90% at negotiation;
history said 48%. Calibration alone removes most of the error. Then give a range (commit / most likely / best case),
not one number.

**4. A sales leader says rep X is the best closer. How do you check?**
Put a confidence interval on every rep's win rate and test whether the spread is larger than chance. In CastIQ the spread
looked like 45%–50%, but p = 1.00 — it was noise.

**5. How do you handle bad CRM data?**
Automate the checks, give each one a severity and an owner, and report the trend weekly. 14 checks in CastIQ;
for example 78% of contacts had a seniority that contradicted their title — that breaks persona routing.

**6. Why should we hire a salesperson for a RevOps / GTM engineering role?**
Because I know what a rep does with a bad score or an unfair territory: ignores it. I design the system so that the
people who use it believe it — transparent rules, intervals on every number, and fields that match how buyers really buy.

# Experiment design: system-chosen geography in large-area search

| | |
|---|---|
| **Status** | Design only. No traffic data exists for this system. |
| **Surface** | Search resultsfor large-areas (no neighbourhood specified) |
| **Change** | The candidate set narrows to *k* system-chosen neighbourhoods before ranking (two arms: *k* = 2, *k* = 3) |
| **Primary metric** | Booking-intent rate per session |
| **Proposed duration** | 4 weeks, fixed horizon |

---

## 1. Summary

Guests searching a whole city rather than a neighbourhood get a candidate set an order of
magnitude larger than a scoped search, and a ranker with no signal about *where* they want to stay
must order it on listing quality alone. This experiment tests whether narrowing that set to a small
number *k* of system-chosen neighbourhoods, before ranking, produces a better first screen than ranking
the whole city. Two variants will be tested, *k* = 2 and *k* = 3, so a null at one dose cannot be mistaken for a
null on the idea.

The hypothesis is falsifiable, the ranking model is identical in every arm, and offline simulation
has already established the cost side of the trade: narrowing is quality-negative below three
neighbourhoods and quality-neutral at five (Appendix A). What offline work **cannot** establish is
the benefit side: whether the narrowed set matches where the guest actually wanted to go. That
question needs users, and it is the reason to run the test.

---

## 2. Background

The system has **one learned stage**. The candidate set is assembled by a *filter* — the query-group
key, plus the geographic narrowing policy this experiment puts under test. Then our LambdaMART model
over 61 listing features orders it and returns the top *N*. The ranker was evaluated offline at
NDCG@10 = 0.7530 [0.7148, 0.7903] on held-out data, against 0.6429 for a price-and-rating heuristic
and a 0.5519 random floor.

The ranker holds no information about the guest. It has no dates, no party history, no prior
sessions, no notion of destination preference. It orders listings by expected demand within
whatever candidate set it is handed. For a search that already names a neighbourhood this is
enough, as the guest has supplied the geography. For a large-area search it is not, because the
system must decide *which parts of the city to show* and the ranker has no basis for that decision
beyond listing quality.

---

## 3. Hypothesis

> **H₁.** For large-area searches, restricting the candidate set to the *k* neighbourhoods with the
> highest historical demand, then ranking within them, increases the booking-intent rate relative
> to ranking the whole area.
>
> **H₀.** Booking-intent rate is unchanged.

Directional secondary prediction, registered in advance:

> **H₂.** At small *k*, narrowing increases the share of first-screen slots held by high-quality
> never-reviewed listings, because new inventory competes locally rather than against an entire
> city. The offline prediction is **differential across the arms**: an increase at *k* = 2
> (0.067 vs 0.048, roughly +40 %) and no change at *k* = 3 (0.049 vs 0.048, Appendix A).
> Observing the increase in one arm and its absence in the other, on the same traffic in the same
> window is a sharper test than either prediction alone.

H₂ is registered because it is measurable, was predicted before the test, and runs *counter* to the
usual expectation that a change favouring established listings will worsen cold-start exposure. An
increase in the *k* = 3 arm would mean live dynamics — reviews earned during the window compounding
exposure — exceed what the static simulation captures.

---

## 4. What changes

### 4.1 Scope

**Eligible:** searches specifying a city, a room type and a party size, but **not** a
neighbourhood. Scoped searches are unaffected and unassigned.

### 4.2 Arms

![How traffic is split: eligibility, assignment, the three candidate-set policies, the one shared ranker, and the two comparisons](figures/ab_traffic_split.png)

| | candidate set | ranking |
|---|---|---|
| **Control** | the whole eligible area | unchanged model, top 10 shown |
| **Treatment *k* = 2** | the 2 highest-prior neighbourhoods only | unchanged model, top 10 shown |
| **Treatment *k* = 3** | the 3 highest-prior neighbourhoods only | unchanged model, top 10 shown |

**The ranking model is byte-identical in all three arms.** Only the candidate set differs, which is
what makes any measured effect attributable to the narrowing policy rather than to the model.

### 4.3 The demand prior

A score per neighbourhood: mean realised demand over a trailing window. It is a **prior over
geographies** and it enters only as a filter on the
candidate set. A neighbourhood with no history falls back to its city's mean rather than being
excluded, so a thin neighbourhood is treated as average instead of vanishing from every search.

Two guards on the score, both declared before launch. The mean is **shrunk toward the city mean**
in proportion to the neighbourhood's inventory — the same empirical-Bayes form as the ranker's
`rating_shrunk` — so a dozen hot listings cannot buy a top-*k* slot on one noisy month. And a
neighbourhood qualifies for the top *k* only above a **minimum-inventory floor**: a narrowing
policy keeping two neighbourhoods must keep enough eligible listings to fill a first screen many
times over, or the narrowed candidate set is a short list wearing a policy.

It is derived from the outcome, so it must be fitted on a window disjoint from the measurement
period. A prior fitted on the traffic it is being evaluated against leaks the outcome into the arm
under test.

**Two doses, *k* = 2 and *k* = 3.** Appendix A rules out the rest: *k* = 1 costs a third of a
grade point and eleven points of relevant share; *k* = 5 is indistinguishable from control and
would measure nothing. Between them the doses divide the labour. Three is quality-neutral offline
(a cost under one-tenth of a grade point), so any intent gain there is nearly free — but its
offline cold-start gain has already decayed to noise (+0.001 on 23 searches). Two carries a real
quality cost (0.10 grade points) and a real predicted cold-start gain: the strongest tolerable
dose. A single dose would confound *narrowing does not pay* with *this k does not pay*; two arms
let the comparison locate where the destination-match benefit crosses the quality cost.

### 4.4 What does not change

The ranking model, its features, the card layout, pricing display, and the treatment of scoped
searches. Any of these moving during the window invalidates the result (§10).

---

## 5. Metrics

### 5.1 Primary

**Booking-intent rate per eligible session** — the proportion of assigned sessions with at least
one eligible search in which the guest reaches date selection and initiates a reservation.

Per session, not per search, so the metric's denominator is the randomisation unit of §6 and the
sample sizes of §7 apply to it directly. A per-search rate under session-level assignment is a
ratio metric whose variance needs the delta method or a session-level bootstrap; the primary
should not owe a correction.

It is the last step the ranker can be said to have caused, and the first that means what the system
was built to mean.

*Not click-through:* a click metric rewards surfacing cheap and photogenic inventory, and the
result card already displays rating, review count and price, so clicks would track the model's
inputs for reasons unrelated to its contribution. Retained as a **leading indicator** only,
readable within days, never the decision.

*Not confirmed bookings:* host acceptance sits between the ranker and the outcome and is not
randomised. The system changes who is *asked*, not who *accepts*.

### 5.2 Secondary

| metric | why | prediction |
|---|---|---|
| deserving cold-start share of first screen | H₂ | **increase** in the *k* = 2 arm; **no change** in the *k* = 3 arm |
| listing-detail views per eligible search | mechanism check between impression and intent | increase |
| distinct neighbourhoods in the first screen | descriptive; falls by construction | **decrease** |
| top-3 click-through | leading indicator | — |

Note the third: the treatment **reduces** per-screen geographic diversity by design, since the
screen cannot show more neighbourhoods than the narrowing kept. It is reported for interpretation, not
as a guardrail — a guardrail on a quantity the treatment necessarily moves would fail every time.

### 5.3 Guardrails

| guardrail | rationale | action on breach |
|---|---|---|
| **neighbourhood exposure coverage** — share of neighbourhoods receiving any first-screen slot, computed per treatment arm over its eligible searches plus all scoped traffic (the ship state — pooled coverage is propped up by the control arm for exactly as long as the test runs) | the supply-side risk the treatment creates: a demand prior can starve low-prior neighbourhoods of traffic entirely, which is invisible in every guest-side metric and compounds, since starved inventory generates no demand signal for the next window | **ship-blocking** |
| **cold-start exposure** — deserving never-reviewed listings reaching the first screen | on this surface the unnarrowed ranking already buries them — 2.5 % reach against a 4.6 % random reference (Appendix A; 6.2 % vs 10.9 % is the neighbourhood-scoped figure) — and a live burial is self-reinforcing within the window, because buried listings earn no reviews and stay buried | **ship-blocking on degradation** |
| median price of the reserved listing | detects a policy winning by surfacing cheap inventory | investigate |
| search abandonment (no listing opened) | detects a screen that got worse in a way intent does not register | investigate |
| host acceptance rate | if the treatment changes who is asked and acceptance falls, intent gains do not convert | investigate |
| sample ratio | assignment integrity | halt |

---

## 6. Experimental design

**Unit: the session.** Assignment is a stable hash of the session id, so a guest never sees more
than one policy within a visit, and map or filter interactions that re-issue the query inherit the
session's arm. Split into equal thirds — a √2-weighted control (41/29/29) is the textbook optimum
for two comparisons sharing one control, but it buys a few percent of variance and complicates the
sample-ratio check, so equal counts are declared instead.

**Interference is present and is not designed away.** Listings are shared inventory: a treatment
guest booking a listing removes it from what a control guest can book. This violates the stable
unit treatment value assumption in a way session-level randomisation cannot fix. Three
consequences, stated rather than corrected:

1. If the treatment works, its bookings deplete exactly the inventory control would otherwise
   have surfaced, depressing control's rate — so the measured difference is **inflated away from
   the null**, the standard direction for demand-side marketplace tests over shared inventory
   (Blake & Coey, 2014). The bias scales with the true effect: a genuinely null policy suffers
   none of it, which is what keeps a tight null interval interpretable (§9).
2. The test also understates ship-state congestion: under a three-way split only a third of the
   eligible traffic runs each narrowed policy, so a treatment arm books against inventory a full
   rollout would consume. Both forces point the same way — **the shipped effect will be smaller
   than the measured one.**
3. The independence assumption behind the standard variance estimator fails, so the nominal
   interval is narrower than the true one.

Session-hash assignment carries a further cost: a returning guest can cross arms between visits,
diluting any effect that accumulates across visits, and every analysis that needs a guest identity
— the novelty exclusion, the repeat/first-time segment, the CUPED covariate — applies only to the
share of sessions linkable to a guest. The design accepts this because search traffic cannot be
assumed logged-in; where a durable guest identifier exists, randomise on it instead.

A switchback design, alternating policies in time within a market, is the standard remedy for the
interference. It is not the launch design because it trades away user-level precision this test can
ill afford (§7) — and "run it if the result is ambiguous" is not a rule, so the trigger is declared
now: the switchback follow-up runs if a significant primary increase coincides with control's
in-window rate falling below its pre-period forecast interval — the depletion signature of §10 —
because that pair cannot be split between true effect and cannibalisation by any within-window
analysis. A null does not trigger it: the interference bias scales with the true effect, so a null
is not concealing one.

---

## 7. Sample size and duration

Two-proportion test per comparison, power 0.80, baseline booking-intent rate **4.0 %**. The table
below is the single-comparison lookup at α = 0.05 two-sided; the launched design runs **two**
comparisons against a shared control at α = 0.025 apiece (§8), which multiplies every per-arm
entry by **1.21**.

| MDE (relative) | absolute | sessions per arm |
|---|---|---|
| +3 % | +0.12 pp | 424,617 |
| +5 % | +0.20 pp | 154,302 |
| +8 % | +0.32 pp | 61,116 |
| +10 % | +0.40 pp | 39,473 |

Duration follows from eligible traffic, which no measurement in this project constrains:

| eligible sessions/day (all three arms) | +5 % MDE | +8 % MDE |
|---|---|---|
| 4,000 | 20.0 weeks | 7.9 weeks |
| 8,000 | 10.0 weeks | 4.0 weeks |
| 20,000 | 4.0 weeks | 1.6 weeks |

**Target: +5 % relative per comparison, 4-week fixed horizon** — three arms at α = 0.025 per
comparison, roughly 187,000 sessions per arm — which requires roughly 20,000 eligible sessions per
day across the three arms; at 11,000 a day the same target takes about 7.3 weeks. Below the
assumed traffic, either the horizon extends or the detectable effect grows, and the table above is
the lookup.

Applying CUPED with pre-period guest activity as the covariate typically reduces the requirement
by 20–50 % on a metric with a stable pre-period. It is margin in the 4-week target, not a
requirement for it — the 20,000-a-day figure is computed without it — and it helps only the
sessions linkable to a guest with history (§6), so the anonymous share of traffic earns none of it.

**Minimum two full weekend cycles regardless of power**, since weekday and weekend search behaviour
differ and a run ending mid-week measures a biased slice of demand.

*Market-size sanity check.* The catalogue supports on the order of 1,500 confirmed stays per day
across the three cities, inferred from 375,806 reviews in a twelve-month window at a 50–72 % review
rate. This bounds the market; it does not determine search volume, which depends on a conversion
rate this project cannot observe.

---

## 8. Analysis plan

**Estimator.** Difference in booking-intent rate between each treatment arm and control, computed
on sessions assigned, not sessions exposed — intention to treat. Sessions that were assigned but issued no eligible search are
excluded at the eligibility filter, which is applied identically to all arms and before assignment
is read.

**Interval.** Two-proportion normal-approximation interval, with the caveat of §6 attached whenever
it is quoted. Bootstrap over sessions as a cross-check.

**Variance reduction.** CUPED on pre-period guest activity, with the adjustment coefficient
estimated on the pre-period and frozen before the analysis date. Applies only to sessions linkable
to a guest (§6); anonymous sessions enter unadjusted.

**Multiplicity.** The primary metric is tested twice — each dose against the shared control — at
α = 0.025 apiece (Bonferroni; Dunnett's procedure, which exploits the correlation induced by the
shared control, is the marginally less conservative refinement if declared before launch). The
*k* = 2 versus *k* = 3 contrast is estimated with an interval, never tested — it selects the ship
candidate in §9, it does not gate shipping. Secondary metrics are reported with Benjamini–Hochberg
control at 0.05 across the eight arm-level results (four metrics × two arms). Guardrails are **not**
multiplicity-corrected — they are one-sided checks whose purpose is to catch harm, and correcting
them raises the bar for detecting exactly what they exist to detect.

**Segments, registered in advance.** Repeat versus first-time guests. The treatment substitutes a
system-chosen destination for the guest's own, so its effect should differ between guests who know
where they want to go and guests who do not. Registering the segment before launch prevents a
post-hoc split from being read as a hypothesis. Prior interactions are **not a confound** —
randomisation balances them — they are a heterogeneity axis and the natural CUPED covariate.

**Novelty.** Reported a second time with each guest's first week of exposure excluded, as a
pre-declared sensitivity rather than a rescue.

**No interim analysis.** The analysis date is fixed before launch. Repeatedly testing an
accumulating sample inflates the false-positive rate far above the nominal level. If continuous
monitoring is required it must use a named sequential procedure — an alpha-spending function or an
always-valid confidence sequence — chosen in advance.

---

## 9. Decision rules

Declared before launch. The point of writing them down is that no case is decided after seeing the
result.

| result | guardrails | decision |
|---|---|---|
| at least one arm significantly up | held in that arm | **Ship the winning arm** — the larger point estimate when both clear, *k* = 3 when their intervals make them indistinguishable, as the quality-safer dose. Expect the shipped gain to undershoot the measured one (§6). |
| an arm significantly up | ship-blocking guardrail degraded in that arm | **Do not ship that arm.** Neighbourhood starvation and cold-start burial both compound over time, and a first-screen gain does not pay for either. If the other arm is up with its guardrails held, ship it; otherwise re-run with an exposure floor in the narrowing policy. |
| neither arm significant | all held | **Do not ship**, and record both intervals. Interference bias scales with the true effect (§6), so tight nulls centred on zero are close to unbiased and are strong evidence the policy does not pay; wide ones mean the test was underpowered and are reported as such rather than as evidence of no effect. |
| *k* = 2 significantly down, *k* = 3 not | any | The quality cost bites before the destination benefit at two neighbourhoods. *k* = 3's own row decides shipping; the ordering across doses is itself the dose–response finding. |
| *k* = 3 significantly down | any | **Abandon.** At the quality-neutral dose a decrease means system-chosen geography is worse than letting the guest browse the whole city — the narrowing hypothesis is falsified — and *k* = 5 is already indistinguishable from control, so there is no parameter left to search. |

---

## 10. Risks

| risk | detection | mitigation |
|---|---|---|
| interference inflates the estimate (§6) | control's own metric drifts against its pre-period | treat the estimate as an upper bound; switchback follow-up |
| a co-occurring change to the narrowing policy, the ranking model or the result card | change log reviewed at the analysis date | freeze the surface for the window |
| the demand prior goes stale mid-window | prior recomputed and compared, not applied | fit once, before launch, and hold it fixed |
| eligible traffic below assumption | monitored from day one | extend the horizon or restate the MDE; do not stop early on a favourable reading |
| sample ratio mismatch | daily | halt; nothing downstream is interpretable |
| exposure logging incomplete | assigned-versus-exposed reconciliation | intention-to-treat analysis is robust to it |

---

## Appendix A — Offline evidence

Computed on held-out listings the ranking model never trained on. Reproduce with:

```bash
uv run python -m rental_ranking.evaluate.exposure
```

**The two policies over the same broad searches**, same ranker, first screen of 10. Averaged over
23 held-out large-area searches — a small sample, and treated as directional rather than precise.

| *k* | arm | mean grade | share grade ≥ 3 | distinct neighbourhoods | deserving cold-start share |
|---|---|---|---|---|---|
| — | control | 3.091 | 0.742 | 2.52 | 0.048 |
| 1 | treatment | 2.778 | 0.635 | 1.00 | **0.091** |
| 2 | treatment | 2.990 | 0.715 | 1.52 | 0.067 |
| 3 | treatment | 3.026 | 0.725 | 1.91 | 0.049 |
| 5 | treatment | 3.108 | 0.751 | 2.30 | 0.048 |

Three readings.

**Narrowing costs first-screen quality, and the cost is steep below three neighbourhoods.** At
*k* = 1 the mean grade falls 0.31 and the relevant share falls 11 points. At *k* = 5 the policy is
quality-neutral — and therefore pointless, since it changes nothing.

**Narrowing nearly doubles deserving cold-start exposure at small *k*.** From 0.048 to 0.091. A
good new listing in a chosen neighbourhood competes against that neighbourhood rather than against
an entire city's established inventory. This was not the expected direction and it is why H₂ is
registered. The gain decays fast — by *k* = 3 it is gone (0.049 vs 0.048) — which is why H₂'s
prediction is differential: an increase in the *k* = 2 arm and none in the *k* = 3 arm.

**Narrowing reduces per-screen geographic diversity**, necessarily — the screen cannot show more
neighbourhoods than the narrowing kept. Any intuition that geographic narrowing increases variety is
backwards, and §5.2 treats the metric descriptively for that reason.

### Why the offline result cannot settle the question

Every column above measures **the cost of narrowing**. None measures its benefit, because the
benefit is that the retained neighbourhoods are the ones the guest wanted, and no offline dataset
contains a guest's destination preference. The simulation locates the trade-off and rules out
*k* = 1 and *k* = 5; only the live test can say whether the exchange is worth making.

### Supporting measurements

Candidate-set size as the search widens, whole catalogue:

| candidate set defined by | groups | median | p90 | max |
|---|---|---|---|---|
| city × neighbourhood × room type × capacity | 516 | 14 | 233 | 2,088 |
| city × room type × capacity | 41 | 55 | 3,825 | 8,948 |
| city × room type | 11 | 173 | 13,197 | 23,441 |

Geographic concentration and cold-start reach when the ranker is handed the wider set directly,
with no narrowing:

| candidate set | distinct neighbourhoods in top 10 | coverage of those reachable | cold-start reach | random reference |
|---|---|---|---|---|
| neighbourhood-scoped | 1.00 | 1.00 | 6.2 % | 10.9 % |
| city × room × capacity | 2.52 | 0.73 | 2.5 % | 4.6 % |
| city × room | 3.33 | 0.69 | 0.2 % | 1.0 % |

Reach relative to chance degrades 0.57 → 0.54 → **0.22** as the set widens: an unnarrowed
large-area ranking buries new listings roughly five times worse than chance. This is the strongest
argument for the cold-start guardrail, and part of the motivation for narrowing at all.

**No NDCG appears in this appendix, and the analysis module exposes no function that could produce
one.** NDCG normalises by the candidate set, so it changes meaning the moment the set changes and
cannot compare two narrowing policies. First-screen composition is unnormalised — ten listings are
ten listings — which is precisely what makes it comparable.

---

## Appendix B — Assumptions register

| quantity | value | source |
|---|---|---|
| catalogue size | 44,684 listings | measured |
| reviews, trailing 12 months | 375,806 | measured |
| confirmed stays/day, three cities | ~1,500 | inferred from reviews at a 50–72 % review rate |
| ranking latency | 87 ms for the full catalogue | measured |
| offline first-screen composition | Appendix A | measured on held-out listings |
| cold-start reach vs chance | 6.2 % vs 10.9 % | measured on held-out listings |
| **eligible sessions/day** | unknown | **assumed**; duration is given as a lookup |
| **booking-intent rate** | 4.0 % | **assumed** |
| **α, power** | 0.05 two-sided family, 0.025 per comparison; 0.80 | convention; Bonferroni across the two doses |
| **CUPED reduction** | 20–50 % | **assumed** from typical behaviour; guest-linkable sessions only |

The three assumed quantities in bold drive the duration entirely. None is observable in this
project, and §7 presents duration as a function of the first rather than a single number.

---

## Appendix C — Limitations

1. **No experiment was run.** There is no live product, no traffic, and no behavioural data — no
   clicks, sessions, queries or guests. Sections 3 and 5–10 are design.
2. **The offline simulation rests on 23 large-area searches** in the held-out set. It is
   directional and is used to rule out configurations, not to size the effect.
3. **No mapping exists from offline ranking quality to booking rate.** The ranker's NDCG motivates
   building the system; it cannot predict an online effect, and no number in §7 is derived from it.
4. **The training target is a demand proxy** — forward calendar availability, not booking history.
   A blocked date may be booked, host-closed or withheld, and the data cannot distinguish them.
5. **The target is downstream of the incumbent marketplace's own ranking.** A listing's occupancy
   partly reflects the exposure it was already given, so "orders by expected demand" and "partly
   reproduces the incumbent ordering" cannot be separated with a single public snapshot.
6. **The ranker has no guest-side signal at all.** Destination intent is a distribution over facets
   — geography, dates, price, amenities, room type, new versus established. This system holds the
   listing side of most of them and the guest side of none. It is the ranking a system uses when it
   knows nothing about who is asking, which is a real and substantial share of traffic, and the
   component a personalisation layer would refine rather than replace. The demand prior in §4.3 is
   the crudest possible stand-in for that layer, and this experiment measures whether even the
   crudest version pays.

---

## Appendix D — Instrumentation

Everything in §5–§9 is computable from five events; none of them exists until built, and the
design is not runnable until they do.

| event | logged when | carries | feeds |
|---|---|---|---|
| `assignment` | the first time a session's arm is read — its first eligible search, not session start | session id, arm, *k*, prior version hash, timestamp | the primary's denominator; the sample-ratio check |
| `search` | every search issued | search id, session id, facets (city, room type, party size, neighbourhood or none), eligibility flag | the eligibility filter; abandonment |
| `impression` | a first screen rendered | search id, the ten listing ids in rank order, each with its neighbourhood and never-reviewed status | both ship-blocking guardrails; H₂; the diversity metric |
| `detail_view` | a listing opened from results | search id, listing id, slot position | the mechanism secondary; top-3 click-through |
| `intent` | date selection reached and a reservation initiated | session id, listing id, timestamp | the primary; the median-price guardrail |

Four notes.

**Attribution is trivial by construction.** The primary is per session, so any `intent` event in an
assigned session counts — no last-touch rule across a session's searches is needed. That is a
second, unadvertised benefit of defining the metric per session (§5.1).

**The policy is logged, not assumed.** `assignment` carries the prior's version hash and *k*, so a
mid-window prior refresh or dose change is detectable in the data rather than only in a change log
(§10).

**Guest linkage is a separate, optional table.** A session-id-to-guest-id mapping where the guest
is logged in is what CUPED, the novelty exclusion and the repeat/first-time segment consume; every
one of them degrades gracefully to the linkable share and no further (§6).

**Three jobs run daily from day one:** the sample-ratio check over `assignment`; the
assigned-versus-exposed reconciliation (§10); and the coverage guardrail over each treatment arm's
impressions plus scoped traffic (§5.3).

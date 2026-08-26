# Experiment design: system-chosen geography in large-area search

| | |
|---|---|
| **Status** | Design only. No traffic data exists for this system. |
| **Surface** | Search results for large-area queries, where no neighbourhood is specified |
| **Change** | The candidate set narrows to *k* system-chosen neighbourhoods before ranking (two arms: *k* = 2, *k* = 3) |
| **Primary metric** | Booking-intent rate per session |
| **Proposed duration** | 4 weeks, fixed horizon |

---

## 1. Summary

Guests searching a whole city rather than a neighbourhood get a candidate set an order of magnitude
larger than a scoped search, and a ranker with no signal about *where* they want to stay must order
it on listing quality alone. This experiment tests whether narrowing that set to a small number *k*
of system-chosen neighbourhoods, before ranking, produces a better first screen than ranking the
whole city. Two variants are tested, *k* = 2 and *k* = 3, so a null at one dose cannot be mistaken
for a null on the idea.

The hypothesis is falsifiable, the ranking model is identical in every arm, and offline simulation
has already established the cost side of the trade: narrowing is quality-negative below three
neighbourhoods and quality-neutral at five (Appendix A). What offline work **cannot** establish is
the benefit side, whether the narrowed set matches where the guest actually wanted to go. That
question needs users, and it is the reason to run the test.

---

## 2. Background

The system has **one learned stage**. The candidate set is assembled by a *filter*: the query-group
key, plus the geographic narrowing policy this experiment puts under test. The LambdaMART model over
61 listing features then orders that set and returns the top 10. The ranker was evaluated offline at
NDCG@10 = 0.7530 [0.7148, 0.7903] on held-out data, against 0.6429 for a price-and-rating heuristic
and a 0.5519 random floor.

The ranker holds no information about the guest. It has no dates, no party history, no prior
sessions, no notion of destination preference. It orders listings by expected demand within whatever
candidate set it is handed. For a search that already names a neighbourhood this is enough, since
the guest has supplied the geography. For a large-area search it is not, because the system must
decide *which parts of the city to show* and the ranker has no basis for that decision beyond
listing quality.

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
> (0.067 against 0.048, roughly +40 %) and no change at *k* = 3 (0.049 against 0.048, Appendix A).
> Observing the increase in one arm and its absence in the other, on the same traffic in the same
> window, is a sharper test than either prediction alone.

H₂ is registered because it is measurable, was predicted before the test, and runs *counter* to the
usual expectation that a change favouring established listings will worsen cold-start exposure. An
increase in the *k* = 3 arm would mean live dynamics, reviews earned during the window compounding
exposure, exceed what the static simulation captures.

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

**Why these two values of *k*.** Appendix A rules out the rest: *k* = 1 costs a third of a grade
point and eleven points of relevant share, while *k* = 5 is indistinguishable from control and would
measure nothing. Between them the two variants divide the labour. Three is quality-neutral offline,
so any intent gain there is nearly free. Two carries a real quality cost, 0.10 grade points, and a
real predicted cold-start gain. A single variant would confound *narrowing does not pay* with *this
k does not pay*, while two arms let the comparison locate where the destination-match benefit
crosses the quality cost.

### 4.3 The demand prior

A score per neighbourhood: mean realised demand over a trailing window. It is a **prior over
geographies**, and it enters only as a filter on the candidate set. A neighbourhood with no history
falls back to its city's mean rather than being excluded, so a thin neighbourhood is treated as
average instead of vanishing from every search.

Two guards on the score. First, the mean is **shrunk toward the city mean** in proportion to the
neighbourhood's inventory, the same empirical-Bayes form as the ranker's `rating_shrunk`, so a dozen
hot listings cannot buy a top-*k* slot on one noisy month. Second, a neighbourhood qualifies for the
top *k* only above a **minimum-inventory floor**: a narrowing policy keeping two neighbourhoods must
keep enough eligible listings to fill a first screen many times over.

The prior is derived from the outcome, so it must be fitted on a window disjoint from the
measurement period. A prior fitted on the traffic it is being evaluated against leaks the outcome
into the arm under test. In production the score would refresh on a cadence, so a neighbourhood that
starts meeting guest demand climbs into the top *k* within weeks; for the duration of this
experiment it is fitted once and frozen, because a prior moving under the arms would confound the
measurement with its own drift (§10).

### 4.4 What does not change

The ranking model, its features, the card layout, pricing display, and the treatment of scoped
searches. Any of these moving during the window invalidates the result (§10).

---

## 5. Metrics

### 5.1 Primary

**Booking-intent rate per eligible session**: the proportion of assigned sessions with at least one
eligible search in which the guest reaches date selection and initiates a reservation. It is the
last step the ranker can be said to have caused.

Per session, not per search, so the metric's denominator is the randomisation unit of §6 and the
sample sizes of §7 apply to it directly.

*Not click-through:* a click metric rewards surfacing cheap and photogenic inventory, and the result
card already displays rating, review count and price, so clicks would track the model's inputs for
reasons unrelated to its contribution.

*Not confirmed bookings:* host acceptance sits between the ranker and the outcome and is not
randomised. The system changes who is *asked*, not who *accepts*.

### 5.2 Secondary

| metric | why | prediction |
|---|---|---|
| deserving cold-start share of first screen | H₂ | **increase** in the *k* = 2 arm; **no change** in the *k* = 3 arm |
| listing-detail views per eligible search | mechanism check between impression and intent | increase |
| top-3 click-through | leading indicator | — |

### 5.3 Guardrails

| guardrail | rationale | action on breach |
|---|---|---|
| **neighbourhood exposure coverage**: share of neighbourhoods receiving any first-screen slot, computed per treatment arm over its eligible searches plus all scoped traffic | the supply-side risk the treatment creates. A demand prior can starve low-prior neighbourhoods of traffic entirely, which is invisible in every guest-side metric and compounds, since starved inventory generates no demand signal for the next window | **ship-blocking** |
| **cold-start exposure**: deserving never-reviewed listings reaching the first screen | on this surface the unnarrowed ranking already buries them, 2.5 % reach against a 4.6 % random reference (Appendix A; 6.2 % against 10.9 % is the neighbourhood-scoped figure), and a live burial is self-reinforcing within the window, because buried listings earn no reviews and stay buried | **ship-blocking on degradation** |
| median price of the reserved listing | detects a policy winning by surfacing cheap inventory | investigate |
| search abandonment (no listing opened) | detects a screen that got worse in a way intent does not register | investigate |
| host acceptance rate | if the treatment changes who is asked and acceptance falls, intent gains do not convert | investigate |
| sample ratio | assignment integrity | halt |

---

## 6. Experimental design

**Unit: the session.** Assignment is a stable hash of the session id, so a guest never sees more
than one policy within a visit, and map or filter interactions that re-issue the query inherit the
session's arm. Traffic splits into equal thirds. A √2-weighted control (41/29/29 % shares) is the
textbook optimum for two comparisons sharing one control, but it buys a few percent of variance and
complicates the sample-ratio check, so equal counts are declared instead.

**Interference is present and is not designed away.** Listings are shared inventory: a treatment
guest booking a listing removes it from what a control guest can book. This violates the stable unit
treatment value assumption (SUTVA) in a way session-level randomisation cannot fix. Three
consequences follow:

1. If the treatment works, its bookings deplete exactly the inventory control would otherwise have
   surfaced, depressing control's rate. The bias scales with the true effect: a genuinely null
   policy suffers none of it, which is what keeps a tight null interval interpretable (§9).
2. The test also understates ship-state congestion. Under a three-way split only a third of the
   eligible traffic runs each narrowed policy, so a treatment arm books against inventory a full
   rollout would consume. Both forces point the same way: **the shipped effect will be smaller than
   the measured one.**
3. The independence assumption behind the standard variance estimator fails, so the nominal interval
   is narrower than the true one.

Session-hash assignment carries a further cost. A returning guest can cross arms between visits,
diluting any effect that accumulates across visits, and every analysis that needs a guest identity,
meaning the novelty exclusion, the repeat/first-time segment and the CUPED covariate, applies only
to the share of sessions linkable to a guest. The design accepts this because search traffic cannot
be assumed logged-in.

A switchback design, alternating policies in time within a market, is the standard remedy for this
interference. It is not the launch design, because it trades away user-level precision this test can
ill afford (§7). But "run it if the result looks odd" is not a rule, so the trigger is declared now:
**the switchback follow-up runs if a significant primary increase coincides with control's in-window
rate falling below its pre-period forecast interval.** That pair is the depletion signature of §10,
and no within-window analysis can split it between a true effect and cannibalisation of control. A
null does not trigger it, because the interference bias scales with the true effect, so a null is
not concealing one.

---

## 7. Sample size and duration

Two-proportion test per comparison, **power 0.80**, **baseline booking-intent rate 4.0 %**. The
table below is the single-comparison lookup at α = 0.05 two-sided. The launched design runs **two**
comparisons against a shared control at α = 0.025 apiece (§8), which multiplies every per-arm entry
by **1.21**.

| MDE (relative) | absolute | sessions per arm |
|---|---|---|
| +3 % | +0.12 pp | 424,617 |
| +5 % | +0.20 pp | 154,302 |
| +8 % | +0.32 pp | 61,116 |
| +10 % | +0.40 pp | 39,473 |

Duration follows from eligible traffic, which no measurement in this project constrains. The weeks
below already carry the 1.21 multiplier, so they are read against the launched two-comparison
design rather than against the table above:

| eligible sessions/day (all three arms) | +5 % MDE | +8 % MDE |
|---|---|---|
| 4,000 | 20.0 weeks | 7.9 weeks |
| 8,000 | 10.0 weeks | 4.0 weeks |
| 20,000 | 4.0 weeks | 1.6 weeks |

**Target: +5 % relative per comparison, 4-week fixed horizon**, which is roughly 187,000 sessions
per arm and requires roughly 20,000 eligible sessions per day across the three arms. At 11,000 a day
the same target takes about 7.3 weeks. Below the assumed traffic, either the horizon extends or the
detectable effect grows.

**Minimum two full weekend cycles regardless of power**, since weekday and weekend search behaviour
differ and a run ending mid-week would measure a biased slice of demand.

*Market-size sanity check.* The catalogue supports on the order of 1,500 confirmed stays per day
across the three cities, inferred from 375,806 reviews in a twelve-month window at a 50–72 % review
rate. That bounds the market. It does not determine search volume, which depends on a conversion
rate this project cannot observe.

---

## 8. Analysis plan

**Estimator.** Difference in booking-intent rate between each treatment arm and control, computed on
sessions assigned. Sessions that were assigned but issued no eligible search are excluded at the
eligibility filter, which is applied identically to all arms and before assignment is read.

**Interval.** Two-proportion normal-approximation interval, with the caveat of §6 attached whenever
it is quoted. Bootstrap over sessions as a cross-check.

**Variance reduction.** CUPED on pre-period guest activity, with the adjustment coefficient
estimated on the pre-period and frozen before the analysis date. It applies only to sessions
linkable to a guest; anonymous sessions enter unadjusted.

**Multiplicity.** The primary metric is tested twice, each dose against the shared control, at
α = 0.025 apiece under a Bonferroni correction. The *k* = 2 against *k* = 3 contrast is estimated
with an interval rather than tested. Secondary metrics are reported with Benjamini–Hochberg control
at 0.05 across the six arm-level results, three metrics × two arms. Guardrails are **not**
multiplicity-corrected.

**No interim analysis.** The analysis date is fixed before launch.

---

## 9. Decision rules

Declared before launch.

| result | guardrails | decision |
|---|---|---|
| at least one arm significantly up | held in that arm | **Ship the winning arm.** Take the larger point estimate when both clear; take *k* = 3 when the intervals make them indistinguishable, since it is the quality-safer option. Expect the shipped gain to undershoot the measured one (§6). |
| an arm significantly up | ship-blocking guardrail degraded in that arm | **Do not ship that arm.** Neighbourhood starvation and cold-start burial both compound over time. |
| neither arm significant | all held | **Do not ship**, and record both intervals. Interference bias scales with the true effect (§6), so tight nulls centred on zero are close to unbiased and are strong evidence the policy does not pay; wide ones mean the test was underpowered. |
| *k* = 2 significantly down, *k* = 3 not | any | The quality cost bites before the destination benefit at two neighbourhoods. *k* = 3's own row decides shipping. |
| *k* = 3 significantly down | any | **Abandon.** At the quality-neutral arm, a decrease means system-chosen geography is worse than letting the guest browse the whole city. |

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

**The two policies over the same broad searches**, same ranker, first screen of 10. Averaged over 23
held-out large-area searches, which is a small sample, so the table is treated as directional.

| *k* | arm | mean grade | share grade ≥ 3 | distinct neighbourhoods | deserving cold-start share |
|---|---|---|---|---|---|
| — | control | 3.091 | 0.742 | 2.52 | 0.048 |
| 1 | treatment | 2.778 | 0.635 | 1.00 | 0.091 |
| 2 | treatment | 2.990 | 0.715 | 1.52 | 0.067 |
| 3 | treatment | 3.026 | 0.725 | 1.91 | 0.049 |
| 5 | treatment | 3.108 | 0.751 | 2.30 | 0.048 |

Three readings:

**Narrowing costs first-screen quality.** At *k* = 1 the mean grade falls 0.31 and the relevant
share falls 11 points. At *k* = 5 the policy is quality-neutral and therefore pointless, since it
changes nothing.

**Narrowing nearly doubles deserving cold-start exposure at small *k*.** The share runs 0.048 at
control to 0.067 at *k* = 2 and 0.091 at *k* = 1, and is gone by *k* = 3 (0.049). A good new listing
in a chosen neighbourhood competes against that neighbourhood rather than an entire city's
established inventory. This is the H₂ prediction, and the reason it is differential across the two
launched arms.

**Narrowing reduces per-screen geographic diversity.** Distinct neighbourhoods on the first screen
fall from 2.52 under control to 1.91 at *k* = 3 and 1.52 at *k* = 2. That is mechanical rather than
surprising, but it is the guest-visible half of what the neighbourhood-coverage guardrail watches on
the supply side.

### Why the offline result cannot settle the question

Every column above measures **the cost of narrowing**, and none measures its benefit. The benefit is
that the retained neighbourhoods are the ones the guest wanted, and **no offline dataset contains a
guest's destination preference**. The simulation locates the trade-off and rules out *k* = 1 and
*k* = 5. Only live traffic can say whether the exchange is worth making.

### Supporting measurements

Candidate-set size as the search widens, whole catalogue:

| candidate set defined by | groups | median | p90 | max |
|---|---|---|---|---|
| city × neighbourhood × room type × capacity | 516 | 14 | 233 | 2,088 |
| city × room type × capacity | 41 | 55 | 3,825 | 8,948 |
| city × room type | 11 | 173 | 13,197 | 23,441 |

Geographic concentration and cold-start reach when the ranker is handed the wider set directly, with
no narrowing:

| candidate set | distinct neighbourhoods in top 10 | coverage of those reachable | cold-start reach | random reference |
|---|---|---|---|---|
| neighbourhood-scoped | 1.00 | 1.00 | 6.2 % | 10.9 % |
| city × room × capacity | 2.52 | 0.73 | 2.5 % | 4.6 % |
| city × room | 3.33 | 0.69 | 0.2 % | 1.0 % |

Reach relative to chance degrades 0.57 → 0.54 → **0.22** as the set widens: an unnarrowed large-area
ranking buries deserving new listings roughly five times worse than chance. This is the strongest
argument for the cold-start guardrail, and part of the motivation for narrowing at all.

---

## Appendix B — Assumptions register

| quantity | value | source |
|---|---|---|
| catalogue size | 44,684 listings | measured |
| reviews, trailing 12 months | 375,806 | measured |
| confirmed stays/day, three cities | ~1,500 | inferred from reviews at a 50–72 % review rate |
| ranking latency | 87 ms for the full catalogue | measured |
| offline first-screen composition | Appendix A | measured on held-out listings |
| cold-start reach vs chance | 6.2 % against 10.9 % | measured on held-out listings |
| **eligible sessions/day** | unknown | **assumed**; duration is given as a lookup |
| **booking-intent rate** | 4.0 % | **assumed** |
| **α, power** | 0.05 two-sided, 0.025 per comparison; 0.80 | convention; Bonferroni across the two doses |
| **CUPED reduction** | 20–50 % | **assumed** from typical behaviour; guest-linkable sessions only |

The three assumed quantities in bold drive the duration entirely. None is observable in this
project, and §7 presents duration as a function of the first.

---

## Appendix C — Limitations

1. **No experiment was run.** There is no live product, no traffic, and no behavioural data: no
   clicks, sessions, queries or guests. Sections 3 and 5–10 are design.
2. **The offline simulation rests on 23 large-area searches** in the held-out set. It is directional
   and is used to rule out configurations, not to size the effect.
3. **No mapping exists from offline ranking quality to booking rate.** The ranker's NDCG motivates
   building the system; it cannot predict an online effect, and no number in §7 is derived from it.
4. **The training target is a demand proxy**: forward calendar availability, not booking history. A
   blocked date may be booked, host-closed or withheld, and the data cannot distinguish them.
5. **The ranker has no guest-side signal at all.** No dates, no history, no destination preference,
   which is the gap this experiment exists to probe rather than a limitation it can remove.

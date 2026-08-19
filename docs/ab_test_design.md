# Experiment design — system-chosen geography in large-area search

| | |
|---|---|
| **Status** | Design. **Not run.** No live traffic exists for this system. |
| **Surface** | Search results, large-area (no neighbourhood specified) |
| **Change** | Retrieval narrows to *k* system-chosen neighbourhoods before ranking |
| **Primary metric** | Booking-intent rate per search |
| **Proposed duration** | 4 weeks, fixed horizon |

---

## 1. Summary

Guests searching a whole city rather than a neighbourhood get a candidate set an order of
magnitude larger than a scoped search, and a ranker with no signal about *where* they want to stay
must order it on listing quality alone. This experiment tests whether narrowing that set to a small
number of system-chosen neighbourhoods, before ranking, produces a better first screen than ranking
the whole city.

The hypothesis is falsifiable, the ranking model is identical in both arms, and offline simulation
has already established the cost side of the trade — narrowing is quality-negative below three
neighbourhoods and quality-neutral at five (Appendix A). What offline work **cannot** establish is
the benefit side: whether the narrowed set matches where the guest actually wanted to go. That
question needs users, and it is the reason to run the test.

---

## 2. Background

The system serves search in two stages. **Retrieval** assembles a candidate set from the catalogue;
**ranking** orders it with a LambdaMART model over 61 listing features and returns the top *K*.
The ranker is evaluated offline at NDCG@10 = 0.7530 [0.7148, 0.7903] on held-out data, against
0.6429 for a price-and-rating heuristic and a 0.5519 random floor.

The ranker holds no information about the guest. It has no dates, no party history, no prior
sessions, no notion of destination preference. It orders listings by expected demand within
whatever candidate set it is handed. For a search that already names a neighbourhood this is
enough — the guest has supplied the geography. For a large-area search it is not, because the
system must decide *which parts of the city to show* and the ranker has no basis for that decision
beyond listing quality.

Ranking a whole city is not a compute problem. The full catalogue of 44,684 listings scores in
**87 ms**; 23,441 in 48 ms. The constraint is informational, not computational: without a view on
destination, a city-wide ranking concentrates the first screen wherever the model's features score
highest, which need not be where the guest wants to be.

---

## 3. Hypothesis

> **H₁.** For large-area searches, restricting the candidate set to the *k* neighbourhoods with the
> highest historical demand, then ranking within them, increases the booking-intent rate relative
> to ranking the whole area.
>
> **H₀.** Booking-intent rate is unchanged.

Directional secondary prediction, registered in advance:

> **H₂.** The treatment increases the share of first-screen slots held by high-quality
> never-reviewed listings, because narrowing makes new inventory compete locally rather than
> against an entire city. Offline simulation predicts roughly a doubling at *k* = 1 (Appendix A).

H₂ is registered because it is measurable, was predicted before the test, and runs *counter* to the
usual expectation that a change favouring established listings will worsen cold-start exposure.

---

## 4. What changes

### 4.1 Scope

**Eligible:** searches specifying a city, a room type and a party size, and **not** a
neighbourhood. Scoped searches are unaffected and unassigned.

### 4.2 Arms

| | retrieval | ranking |
|---|---|---|
| **Control** | the whole eligible area | unchanged model, top 10 shown |
| **Treatment** | the *k* highest-prior neighbourhoods only | unchanged model, top 10 shown |

**The ranking model is byte-identical in both arms.** Only the candidate set differs, which is what
makes any measured effect attributable to the retrieval policy rather than to the model.

### 4.3 The demand prior

A score per neighbourhood: mean realised demand over a trailing window. It is a **prior over
geographies**, never a listing feature — no listing sees it, and it enters only as a filter on the
candidate set. A neighbourhood with no history falls back to its city's mean rather than being
excluded, so a thin neighbourhood is treated as average instead of vanishing from every search.

It is derived from the outcome, so it must be fitted on a window disjoint from the measurement
period. A prior fitted on the traffic it is being evaluated against leaks the outcome into the arm
under test.

**`k` = 3** for the launch configuration. Appendix A locates the trade-off: below three, narrowing
costs first-screen quality; at five it is quality-neutral and buys nothing. Three is where the
cold-start gain is still positive and the quality cost is under one-tenth of a grade point.

### 4.4 What does not change

The ranking model, its features, the card layout, pricing display, and the treatment of scoped
searches. Any of these moving during the window invalidates the result (§10).

---

## 5. Metrics

### 5.1 Primary

**Booking-intent rate per eligible search** — the proportion of eligible searches in which the
guest reaches date selection and initiates a reservation.

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
| deserving cold-start share of first screen | H₂ | **increase** |
| listing-detail views per eligible search | mechanism check between impression and intent | increase |
| distinct neighbourhoods in the first screen | descriptive; falls by construction | **decrease** |
| top-3 click-through | leading indicator | — |

Note the third: the treatment **reduces** per-screen geographic diversity by design, since the
screen cannot show more neighbourhoods than retrieval kept. It is reported for interpretation, not
as a guardrail — a guardrail on a quantity the treatment necessarily moves would fail every time.

### 5.3 Guardrails

| guardrail | rationale | action on breach |
|---|---|---|
| **neighbourhood exposure coverage** — share of neighbourhoods receiving any first-screen slot across all searches | the supply-side risk the treatment creates: a demand prior can starve low-prior neighbourhoods of traffic entirely, which is invisible in every guest-side metric and compounds, since starved inventory generates no demand signal for the next window | **ship-blocking** |
| **cold-start exposure** — deserving never-reviewed listings reaching the first screen | the system already surfaces them at 6.2 % against a 10.9 % random reference; a live burial is self-reinforcing within the window, because buried listings earn no reviews and stay buried | **ship-blocking on degradation** |
| median price of the reserved listing | detects a policy winning by surfacing cheap inventory | investigate |
| search abandonment (no listing opened) | detects a screen that got worse in a way intent does not register | investigate |
| host acceptance rate | if the treatment changes who is asked and acceptance falls, intent gains do not convert | investigate |
| sample ratio | assignment integrity | halt |

---

## 6. Experimental design

**Unit: the session.** Assignment is a stable hash of the session id, so a guest never sees both
policies within a visit, and map or filter interactions that re-issue the query inherit the
session's arm. Split 50/50.

**Interference is present and is not designed away.** Listings are shared inventory: a treatment
guest booking a listing removes it from what a control guest can book. This violates the stable
unit treatment value assumption in a way session-level randomisation cannot fix. Two consequences,
both stated rather than corrected:

1. The measured effect is **attenuated toward the null**, so the estimate is a lower bound and a
   non-significant result is weaker evidence of no effect than it appears.
2. The independence assumption behind the standard variance estimator fails, so the nominal
   interval is narrower than the true one.

A switchback design, alternating policies in time within a market, is the standard remedy and is
the recommended follow-up if the session-level result is ambiguous. It is not the launch design
because it trades away user-level precision that this test can ill afford (§7).

---

## 7. Sample size and duration

Two-proportion test, α = 0.05 two-sided, power 0.80, baseline booking-intent rate **4.0 %**.

| MDE (relative) | absolute | sessions per arm |
|---|---|---|
| +3 % | +0.12 pp | 424,617 |
| +5 % | +0.20 pp | 154,302 |
| +8 % | +0.32 pp | 61,116 |
| +10 % | +0.40 pp | 39,473 |

Duration follows from eligible traffic, which no measurement in this project constrains:

| eligible sessions/day (both arms) | +5 % MDE | +8 % MDE |
|---|---|---|
| 4,000 | 11.0 weeks | 4.4 weeks |
| 8,000 | 5.5 weeks | 2.2 weeks |
| 20,000 | 2.2 weeks | 0.9 weeks |

**Target: +5 % relative, 4-week fixed horizon**, which requires roughly 11,000 eligible sessions per
day. Below that, either the horizon extends or the detectable effect grows, and the table above is
the lookup.

Applying CUPED with pre-period session activity as the covariate typically reduces the requirement
by 20–50 % on a metric with a stable pre-period, and is assumed in the 4-week target.

**Minimum two full weekend cycles regardless of power**, since weekday and weekend search behaviour
differ and a run ending mid-week measures a biased slice of demand.

*Market-size sanity check.* The catalogue supports on the order of 1,500 confirmed stays per day
across the three cities, inferred from 375,806 reviews in a twelve-month window at a 50–72 % review
rate. This bounds the market; it does not determine search volume, which depends on a conversion
rate this project cannot observe.

---

## 8. Analysis plan

**Estimator.** Difference in booking-intent rate between arms, computed on sessions assigned, not
sessions exposed — intention to treat. Sessions that were assigned but issued no eligible search are
excluded at the eligibility filter, which is applied identically to both arms and before assignment
is read.

**Interval.** Two-proportion normal-approximation interval, with the caveat of §6 attached whenever
it is quoted. Bootstrap over sessions as a cross-check.

**Variance reduction.** CUPED on pre-period session activity, with the adjustment coefficient
estimated on the pre-period and frozen before the analysis date.

**Multiplicity.** The primary metric is a single pre-declared test at α = 0.05. Secondary metrics
are reported with Benjamini–Hochberg control at 0.05 across the four. Guardrails are **not**
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

Declared before launch. The point of writing them down is that none of the four cases is decided
after seeing the result.

| primary | guardrails | decision |
|---|---|---|
| significant increase | all held | **Ship** at *k* = 3. Follow with a *k* sweep. |
| significant increase | ship-blocking guardrail degraded | **Do not ship.** Neighbourhood starvation and cold-start burial both compound over time, and a first-screen gain does not pay for either. Re-run with an exposure floor in retrieval. |
| no significant difference | all held | **Do not ship**, and record the interval. Given the attenuation of §6 the estimate is a lower bound, so a tight null centred on zero is informative; a wide one means the test was underpowered and should be reported as such rather than as evidence of no effect. |
| significant decrease | any | **Do not ship.** The offline simulation predicts this below *k* = 3; a decrease at *k* = 3 would falsify the narrowing hypothesis and the follow-up is a larger *k*, not a repeat. |

---

## 10. Risks

| risk | detection | mitigation |
|---|---|---|
| interference swamps the effect | control's own metric drifts against its pre-period | switchback follow-up |
| a co-occurring change to retrieval, ranking or the result card | change log reviewed at the analysis date | freeze the surface for the window |
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
registered.

**Narrowing reduces per-screen geographic diversity**, necessarily — the screen cannot show more
neighbourhoods than retrieval kept. Any intuition that geographic narrowing increases variety is
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
cannot compare two retrieval policies. First-screen composition is unnormalised — ten listings are
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
| **α, power** | 0.05, 0.80 | convention |
| **CUPED reduction** | 20–50 % | **assumed** from typical behaviour |

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

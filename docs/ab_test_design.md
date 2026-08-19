# A/B test design — system-chosen geography for large-area search

**No A/B test was run.** There is no live product, no traffic, and no behavioural data of any kind
in this project — no clicks, impressions, sessions, queries or users. This document is a design,
plus the offline analysis that motivates it and bounds what it could learn.

Its organising rule is a hard three-way split. Every quantity below is tagged **[M] measured** from
our own data, **[D] derived** from measured quantities under assumptions stated at the point of
use, or **[H] hypothetical** — convention, or a number that would come from a system that does not
exist. Nothing crosses those lines silently. That discipline is the actual deliverable; the numbers
are downstream of it.

Regenerate the measured figures with:

```bash
uv run python -m rental_ranking.evaluate.exposure
```

---

## 1. The ledger

| quantity | value | class |
|---|---|---|
| ranked listings | 44,684 | **[M]** |
| query groups (cascade) | 393 | **[M]** |
| reviews in the last 12 months | 375,806 | **[M]** |
| confirmed stays/day, all three cities | 1,064 – 2,059 | **[D]** review rate 50–72 %, or 3.5–5.5 nights/stay |
| largest candidate set at `city × room_type` | 23,441 | **[M]** |
| endpoint per-request cap | 5,000 | **[M]** `cloud/score.py` |
| inference latency | 12.9 – 14.5 ms | **[M]** deployment log, 2026-08-19 |
| top-10 geographic coverage, widened key | 0.69 – 0.73 | **[M]** sealed fold |
| deserving cold-start reach @10 | 6.2 % vs 10.9 % shuffle | **[M]** sealed fold |
| searches/day | ~100,000 | **[D]** 1.5 % search→booking |
| qualifying-search share | 4 – 15 % | **[H]** |
| booking-intent rate per search | 4.0 % | **[H]** |
| α, power | 0.05 two-sided, 0.80 | **[H]** convention |

The last four drive the power calculation, and three of them are hypothetical. That is the honest
state of this design and §9 does not pretend otherwise.

---

## 2. The product surface

Nothing about a results page exists in this project, so it has to be fixed before "top-3" means
anything. **[H]** throughout:

```
┌─────────────────────────────┬──────────────┐
│  ▢ ▢ ▢   ← row 1, pos 1-3   │              │
│  ▢ ▢ ▢   ← row 2, pos 4-6   │     MAP      │
│  ▢ ▢ ▢                      │              │
│  ▢ ▢ ▢   ~18 cards, then    │  pan/zoom    │
│           infinite scroll   │  re-queries  │
└─────────────────────────────┴──────────────┘

card: photo · title · room type · capacity · rating ★ · review count · price/night
```

Three consequences. Positions 1–3 are one visual row, so a top-3 metric is a real unit. The card
surfaces **rating, review count and price** — which is close to what our highest-gain features
encode, so click behaviour will correlate with our score partly for reasons that are not the
model's insight. And map pan re-queries, which fragments a session into several searches and
matters for the randomisation unit.

---

## 3. The funnel, and where the ranker's reach ends

```
① search issued → ② retrieval → ③ RANKING ← us → ④ impression → ⑤ click
   → ⑥ engagement → ⑦ booking intent → ⑧ host acceptance → ⑨ stay → ⑩ review
```

The ranker **fully controls ④**, strongly controls ⑤, and by ⑧ is one cause among many — price,
photos, calendar, host responsiveness. It can only change *which listings get the chance*.

**Our label sits at ⑨.** `blocked_fraction_90` is a 90-day, listing-level, all-users aggregate of
forward calendar availability. We train on an outcome measured at the far end of the funnel and
deploy at ③, with no session, no user and no dates in between.

The consequence, stated once: **the training objective and any test metric are different
quantities.** There is no principled mapping from an NDCG lift to a booking-rate lift. Nothing in
§9's effect size is derived from §5 of notebook 04, and it would be dishonest to imply otherwise.

---

## 4. Why this experiment and not the obvious one

The obvious test is *ranker vs incumbent heuristic*. It is the wrong one to design here, because
offline evaluation has already answered it as well as it can be answered — **0.7530 vs 0.6429 on
the sealed fold, paired +0.1139 [0.0795, 0.1534] [M]** — and an online test of it would be
motivated by a number that cannot be translated into the metric it would move.

The Airbnb paper on booking intent (§4.2, *large area search*) points at a better question. Their
mechanism for a broad search is: bounding box → retrieve candidate geos → intent model recommends
*k* destinations → **ranking boosts listings from those geos**. They do not flatten geography for
large-area search; they *narrow* it, per user.

Our neighbourhood-scoped query group is structurally the same move without personalisation. So the
question worth testing is the one offline evaluation structurally cannot answer:

> **Who chooses the geography — the user, or the system?**

- **Control.** The user pins a neighbourhood; we rank within it. Today's system.
- **Treatment.** The user does not pin one. The system ranks neighbourhoods by a demand prior,
  takes the top *k*, and ranks listings within them — the paper's architecture with a
  non-personalised prior standing in for the intent model.

**The prior must be fitted on training folds only.** It is label-derived, and a demand prior fitted
on the evaluation population would leak the target into the arm being measured.

### Why not simply flatten to city

Because our own data rules it out. **[M]**

| key | groups | median set | p90 | max | over the 5,000 cap |
|---|---|---|---|---|---|
| `city × nbhd × room × tier` (today) | 516 | 14 | 233 | 2,088 | 0 ✅ |
| `city × room × tier` | 41 | 55 | 3,825 | 8,948 | 3 ❌ |
| `city × room` | 11 | 173 | 13,197 | **23,441** | 2 ❌ |

A flat city ranking asks for **23,441 listings in one request** — past the endpoint's own
`MAX_LISTINGS = 5000` and roughly a 45 MB payload. The service we deployed cannot serve the naive
arm. That is the paper's "sheer volume" challenge, measured rather than quoted.

### What widening costs, measured

Scored on the **sealed fold only** — the one population the refit model did not fit. This reads the
holdout for *composition*, never for quality: no NDCG, no baseline comparison, no paired test. The
project's two declared performance reads remain spent and closed; this adds no third. **[M]**

| key | groups | median set | distinct geos in top-10 | coverage | entropy | cold-start reach | shuffle | lift |
|---|---|---|---|---|---|---|---|---|
| `city × nbhd × room × tier` | 112 | 8 | 1.00 | 1.00 | — | 6.2 % | 10.9 % | **−4.7 pp** |
| `city × room × tier` | 23 | 43 | 2.52 | 0.73 | 0.57 | 2.5 % | 4.6 % | −2.1 pp |
| `city × room` | 9 | 80 | 3.33 | 0.69 | 0.62 | 0.2 % | 1.0 % | −0.8 pp |

Read three things off it.

**Widening does collapse geography, but not catastrophically.** A widened top-10 reaches about
**69–73 %** of the neighbourhoods it could have, and is **57–62 %** as evenly spread as a perfectly
diverse screen. Real, worth a guardrail, not fatal — which is why the treatment is worth testing
rather than assumed to fail.

**Widening makes the cold-start problem relatively worse.** The absolute reach falls simply because
ten slots out of eighty is a smaller share than ten out of eight. But the *ratio* to the shuffle
reference degrades too: **0.57 → 0.54 → 0.22**. A flat city ranking buries new listings roughly
five times worse than chance. Any large-area arm needs the cold-start guardrail more than the
control does, not less.

**These rungs are plain re-keyings, not the cascade.** `groups.query_group` widens the key only for
groups below the minimum; the table asks the counterfactual "what if the key had been this for
everyone", which is what the treatment arm proposes.

> **A note on what is deliberately absent.** There is no NDCG column above, and
> `evaluate/exposure.py` exposes no function that could produce one. Changing the key changes the
> candidate set, the ideal DCG and the random floor at once, so a rung-1 NDCG beside a rung-3 NDCG
> compares two quantities that merely share a name — and the coarsest rung *is* the grading
> partition, so its grade distribution is fixed by construction and a model could "improve" there
> by doing nothing. A test asserts the rule.

---

## 5. Primary metric

**Booking-intent rate per search** — the share of searches in which the user reaches date selection
and presses Reserve (step ⑦).

Not **top-3 CTR**: the card already shows rating, review count and price, so clicks would correlate
with our score for reasons that are not the model's contribution, and a click metric rewards a
ranker that surfaces cheap, photogenic, badly-located inventory. It remains a **leading indicator**,
readable in days, never the decision.

Not **confirmed bookings**: host acceptance (⑦→⑧) sits between the ranker and the outcome and is
not randomised. The ranker changes who is *asked*, not who *accepts*.

Booking intent is the last step the ranker can still be said to have caused, and the first that
means what the model was built to mean.

---

## 6. Eligibility — and why it is a constraint, not a choice

**Our scores are only comparable within a query group.** LambdaMART optimises ordering inside a
group and the scores carry no cross-group calibration; `cloud/score.py` states this as its
contract. There is therefore **no merge policy** available to us: two groups' rankings cannot be
interleaved into one list, because doing so compares scores that were never calibrated against each
other.

So the control arm can only serve a search whose candidate set *is* one query group — one that pins
a neighbourhood, a room type and a party size. A real search ("Athens, 12–15 Sept, 2 guests") spans
44 neighbourhoods and cannot be served at all.

Qualifying share is **[H]**, and the power calculation is shown across 4–15 % rather than fixed. It
is also the honest deployment cost of the Phase 2 grouping decision, and it belongs in the README's
limitations as much as here.

---

## 7. Randomisation, and the interference we cannot design away

**Unit: the session**, hashed to an arm on a stable key, with map-pan re-queries inheriting the
session's assignment so one user never sees two rankings.

**Shared inventory breaks SUTVA.** Surfacing listing X to more treatment users fills X's calendar,
which removes X from what control users can book. Treatment changes control's experience. Two
consequences, both real: the measured effect is **attenuated toward zero**, so a null is weaker
evidence of no effect than it looks; and the independence assumption behind the naive variance
estimate fails, so the nominal interval is too narrow.

The alternatives are worse here, and the doc should say so rather than pretend the problem is
solved:

- **Listing-side randomisation** (assign inventory, not users) removes the cannibalisation but
  cannot randomise a *ranking*, which is a property of the whole result set.
- **Market-level (cluster) randomisation** contains interference inside a market, but we have
  **three cities [M]**. Three clusters cannot support inference; even 75 neighbourhoods would be a
  strained cluster count with heavy imbalance.
- **Switchback** (alternate arms in time within a market) is the standard marketplace answer and is
  the right recommendation if the effect is expected to be fast-acting. It trades user-level
  precision for interference containment and needs a carryover assumption stated explicitly.

**Recommendation:** session-level, with the attenuation recorded as a known bias *direction* — the
estimate is a lower bound on the true effect — and a switchback named as the follow-up if the
session-level result is ambiguous. Do not report an interference-free interval as if it were one.

---

## 8. Guardrails

Two of these can stop a ship on their own.

| guardrail | why | threshold |
|---|---|---|
| **cold-start top-10 reach** | **[M]** the ranker already surfaces deserving never-reviewed listings at 6.2 % against a 10.9 % shuffle, and widening the key makes the ratio worse. In a live test that is self-fulfilling *inside the window*: buried listings get no bookings, so no reviews, so stay buried | ship-blocking if it degrades |
| **geographic coverage of top-10** | **[M]** the treatment's own failure mode — 0.69–0.73 coverage offline | ship-blocking if it degrades |
| median price of booked listing | catches a ranker that wins by surfacing cheap inventory | monitor |
| search abandonment (zero clicks) | catches a results page that got worse in a way intent does not see | monitor |
| host acceptance rate | the treatment changes *who* is asked; if acceptance drops, intent gains are illusory | monitor |
| sample-ratio mismatch | assignment integrity | halt on failure |

The first is not a formality. It is the project's known pathology, measured with a random control,
and a test that improved booking intent while worsening it should not ship on the primary metric
alone.

---

## 9. Power — and the design's own verdict on itself

Traffic is **derived from our own data** rather than invented, which is the one part of this
section that is not guesswork. **[M]** 375,806 reviews across the ranked population in twelve
months implies **1,430–2,059 confirmed stays/day** at a 50–72 % review rate; Inside Airbnb's own
occupancy estimate implies **1,064–1,672** at 3.5–5.5 nights per stay. **These two routes are not
independent** — Inside Airbnb's estimate is itself reviews-based — so their agreement checks the
parameters, not the method. Take **~1,500 stays/day [D]**.

At a 1.5 % search→booking rate **[H]** that is **~100,000 searches/day [D]**, and after the
eligibility haircut of §6:

| qualifying share | searches/day | per arm/day |
|---|---|---|
| 4 % | 4,000 | 2,000 |
| 8 % | 8,000 | 4,000 |
| 15 % | 15,000 | 7,500 |

Two-proportion, baseline intent 4.0 % **[H]**, α 0.05 two-sided, power 0.80:

| MDE (relative) | n per arm | at 4,000/arm/day |
|---|---|---|
| +3 % | 424,617 | **15.2 weeks** |
| +5 % | 154,302 | **5.5 weeks** |
| +8 % | 61,116 | **2.2 weeks** |
| +10 % | 39,473 | 1.4 weeks |
| +15 % | 17,940 | 0.6 weeks |

**The verdict this design reaches about itself: at plausible traffic it is underpowered for the
effect it is likely to produce.** A ranking change that moves booking intent by 8 % relative would
be a very large win; the effect worth detecting is nearer 3–5 %, and that needs **6 to 15 weeks** —
long enough that seasonality drift, co-occurring product changes and accumulated interference
become first-order problems rather than caveats.

Saying so is the finding. The remedies, in order of cost:

1. **CUPED** on pre-period session activity. Standard, and typically cuts required *n* by 20–50 %
   on a metric with a stable pre-period — enough to bring +5 % from 5.5 weeks toward 3–4.
2. **Widen eligibility** by giving the model a cross-group calibration layer, which raises the
   qualifying share and is worth doing on its own merits (§6).
3. **Accept a larger MDE** and pre-declare that a smaller true effect will be missed — honest, and
   better than an underpowered test reported as "no difference".
4. **Switchback** to buy precision back from the interference budget.

**Recommended operating point:** MDE **+5 % relative**, α 0.05, power 0.80, CUPED applied,
**fixed 4-week horizon** with the analysis date declared before launch — accepting that this rests
on an 8 % eligibility assumption and stating that, at 4 %, the same design detects only +7 %.

---

## 10. Peeking, novelty, duration

**Fixed horizon, analysis date declared before launch.** Repeatedly testing an accumulating sample
inflates the false-positive rate far above the nominal α, and "it looked significant on Tuesday" is
the most common way a ranking test ships a null. If continuous monitoring is wanted, it must be a
**named** sequential procedure — an alpha-spending function or an always-valid confidence sequence
— chosen in advance, not a habit of looking.

**Minimum two full weekend cycles** regardless of what the power calculation permits. Weekday and
weekend search behaviour differ, and a run that ends mid-week measures a biased slice of demand.

**Novelty applies to repeat users only.** Report the primary effect a second time with the first
week of each user's exposure excluded, as a pre-declared sensitivity rather than a post-hoc rescue.

---

## 11. Pre-registered segments, and variance reduction

**Repeat vs first-time users.** The treatment substitutes a system-chosen geography for a
user-chosen one, so its effect should plausibly differ between users who know where they want to go
and users who do not. Register the segment before launch; discovering it afterwards is a multiple
comparison wearing a hypothesis.

**Past interactions are not a confound.** Randomisation balances them in expectation. They are two
other things: a **heterogeneity axis** (above) and the natural **CUPED covariate** (§9).

---

## 12. What would invalidate the test

- **Sample-ratio mismatch** — arms not landing at the intended split means the assignment or the
  logging is broken, and nothing downstream is interpretable.
- **A co-occurring change** to retrieval, pricing display or the card layout during the window.
- **Interference large enough to swamp the effect**, detectable as a drift in control's own metric
  against the pre-period.
- **Impression logging.** Whether the first row was actually *seen* needs client instrumentation
  most deployments approximate. Without it, any position-sensitive metric is really conditioned on
  the user having scrolled — and that denominator is itself affected by treatment.

---

## 13. Limitations

1. **No A/B test was run.** No traffic, no users, no behavioural data. Sections 5 and 7–11 are
   design; only §4's tables are measurements.
2. **Three of the four quantities driving the power calculation are hypothetical** (§1). The
   traffic anchor is derived from our own reviews; the conversion rate, eligibility share and
   baseline intent rate are not.
3. **No mapping exists from NDCG to booking rate.** The offline result motivates building a ranker;
   it cannot size an online effect.
4. **The label is a demand proxy** — forward calendar availability, never bookings — and it is
   **downstream of Airbnb's own ranker**, so "ranks by expected demand" and "partially imitates the
   incumbent" cannot be separated with one public snapshot. Notebook 04 §11 bounds this.
5. **The eligibility constraint is structural**, not a product decision (§6).

### The framing worth keeping

The paper defines guest intent as a preference distribution over facets: geo, date, price,
amenities, categories, room types, trip types, new-vs-existing listings. We hold the **listing side**
of nearly all of them — geo, price, 19 amenity buckets, room type, and new-vs-existing, which is
precisely our cold-start axis. We hold **no date or trip-type signal at all**, and **no user-side
distribution over any facet**.

So: **their intent model is a posterior; ours is the prior.** A ranker with no user signal is
exactly what serves logged-out and first-session traffic — a real and substantial segment, not a
degenerate case. This model is not a broken version of a personalised one. It is the component a
personalisation layer would multiply into, and the honest claim for it is that it ranks a market
segment well in the absence of any information about who is asking.

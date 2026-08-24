"""Is a ranking score comparable across query groups, or only within one?

The question decides what the system can serve. If a score means the same thing in two searches,
a **broad search** — one naming a city, a room type and a party size but no neighbourhood — is
just one search over a wider candidate set, and the served model can rank it. If it does not,
there is no way to merge and the broad search cannot be served at all.

The project asserted the second answer for a month and never measured it (``cloud/score.py``,
``docs/decisions_log.md`` 2026-08-18 and 2026-08-19). Measured on 2026-08-22, development
out-of-fold, over universes of ``city x room_type x capacity_tier`` — where two query groups
differ **only** by neighbourhood, which is exactly the broad search ``docs/ab_test_design.md``
scopes as eligible — it is **wrong for the case that matters**::

    pairs from the same query group      0.6433  [0.6276, 0.6588]   61 cells
    pairs from different query groups    0.6447  [0.6314, 0.6582]   55 cells
    paired difference                   -0.0016  [-0.0122, +0.0092] 54 cells

Indistinguishable. Ordering across neighbourhoods is as good as ordering inside a group. Stable
to +/- 0.0014 across sampling seeds, and ``price`` as a negative control sits at chance both ways
(0.4875 within, 0.5143 cross), so the instrument is not reporting something trivial.

**The error the assertion rested on was conflating calibration with comparability.** A
LambdaMART score is genuinely uncalibrated — it is not a probability and its scale is arbitrary,
so ``cloud/score.py`` is right to return an ordering rather than raw numbers. But ordering two
listings needs only a monotone common function, not a calibrated one, and the model *is* one
function: it takes no query-group input, so there is no per-group parameter that could differ.
The lambdarank objective draws gradients only from within-group pairs, which leaves cross-group
ordering **unconstrained by training** — not destroyed by it. Those are different claims and only
the first is true by construction.

Three decisions make the measurement mean what it says:

* **Pairwise accuracy, not NDCG.** NDCG normalises by the candidate set, so it is a *different
  quantity* at every grouping — which is exactly why ``exposure.py`` exposes no way to compute it
  across schemes. A pair either orders correctly or it does not, and that means the same thing
  whichever two listings are drawn. Being comparable across grouping schemes is the whole
  requirement here, and pairwise accuracy is comparable by construction.
* **Every cell must be fold-pure.** Out-of-fold scores come from four fold models, each with its
  own scale. ``train.py`` notes this is harmless within a query group *because a group lies
  wholly inside one fold* — but a broad universe does not, and only 15 of 39 are fold-pure,
  covering 0.4 % of the pool. So the evaluation cell is ``(universe, fold)``: one model, one
  scale, every row held out. Pooling a universe across folds would compare fold 1's scores with
  fold 3's and call the artefact a finding.
* **Cells are the resampling unit, not pairs.** Pairs drawn inside one cell are massively
  dependent; bootstrapping over them would give an interval an order of magnitude too tight.

**Scope is a hard boundary, not a caution.** A grade is a quartile within ``city x room_type``,
so pooling grades across two grading partitions compares quantiles computed on different
populations — and the raw label is not poolable across cities either, since the between-city
gradient runs opposite to the within-city one (BUILD_GUIDE gotcha #7). The largest set over which
this question is even *defined* is one grading partition. Every rung of ``groups.GROUP_CASCADE``
sits inside one, because the coarsening rule forces it: the rule imposed to protect the target
also happens to make this measurable.

Widening the universe one more rung, to ``city x room_type`` — the grading partition itself,
where groups differ by capacity tier as well — gives a paired difference of +0.0012
[-0.0132, +0.0157] over 24 cells. Also indistinguishable, but **it is not evidence about capacity
tiers specifically**: cross-neighbourhood pairs dominate that draw by count, so the aggregate
mostly restates the rung above. Isolating the cross-tier-only pairs leaves 15 cells and an
interval of [0.4911, 0.6969], which resolves nothing. Stated rather than glossed: what is
measured here is comparability **across neighbourhoods**, and nothing stronger.

Pure transforms, as everywhere in ``evaluate/`` — no I/O and no ``main()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rental_ranking.evaluate.metrics import bootstrap_ci

#: What "correctly ordered" is judged against. The training target, matching
#: ``fairness.DEFAULT_TRUTH`` and for the same reason: measuring against the finer
#: ``blocked_fraction_90`` scores the model for a coarsening it was never asked to undo. Both
#: give the same verdict here.
DEFAULT_TRUTH = "grade"

#: Pairs sampled per cell before filtering to discordant truth. Sampling rather than enumerating
#: because a cell of 2,700 listings holds 3.6 million pairs and the estimate is stable to
#: +/- 0.005 across seeds well below that.
DEFAULT_DRAWS = 60_000

#: Discordant pairs a cell needs before it reports an accuracy. Below this the cell's estimate is
#: noise that the bootstrap would carry at the same weight as a well-populated one.
MIN_PAIRS = 200


def evaluation_cells(universe: pd.Series, fold: pd.Series) -> pd.Series:
    """The fold-pure unit a cross-group comparison must be computed inside.

    See the module docstring: concatenated out-of-fold scores come from four different fitted
    models, and a broad universe spans folds. Keying the cell on both is what puts one score
    scale inside every candidate set.

    Args:
        universe: The broad search a listing belongs to — normally
            ``exposure.rung_labels(frame, key)`` at a cascade rung coarser than the query group.
        fold: Fold id per row.

    Returns:
        String Series aligned to ``universe``.
    """
    return pd.Series(
        [f"{u}|f{f}" for u, f in zip(universe.to_numpy(), fold.to_numpy(), strict=True)],
        index=universe.index,
        name="cell",
    )


def pair_accuracy(
    scores: pd.Series,
    groups: pd.Series,
    truth: pd.Series,
    cells: pd.Series,
    within: bool,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    min_pairs: int = MIN_PAIRS,
) -> pd.Series:
    """Share of discordant pairs the score orders the way the truth does, per cell.

    Args:
        scores: The ranker's score per row.
        groups: Query-group id per row.
        truth: What the ordering is judged against — normally ``grade``.
        cells: Fold-pure cell id per row, from :func:`evaluation_cells`.
        within: ``True`` draws both listings from one query group — the ordering the model was
            trained to produce. ``False`` draws them from **different** groups inside the same
            cell, which is the ordering nothing in training constrained.
        draws: Pairs sampled per cell before filtering.
        seed: Sampling seed. The estimate is stable to about +/- 0.005 across seeds.
        min_pairs: Discordant pairs a cell needs before it reports.

    Returns:
        Float Series indexed by cell. Cells that cannot supply ``min_pairs`` are absent rather
        than present with a noisy value.
    """
    frame = pd.DataFrame(
        {
            "score": scores.to_numpy(),
            "group": groups.to_numpy(),
            "truth": truth.to_numpy(),
            "cell": cells.to_numpy(),
        }
    )
    rng = np.random.default_rng(seed)
    out: dict[object, float] = {}

    for cell, block in frame.groupby("cell", observed=True, sort=False):
        size = len(block)
        if size < 2:
            continue
        score = block["score"].to_numpy()
        value = block["truth"].to_numpy()
        group = block["group"].to_numpy()

        left = rng.integers(0, size, draws)
        right = rng.integers(0, size, draws)
        # Ties in the truth carry no information about ordering and are dropped rather than
        # counted as half-right, which would pull every estimate toward 0.5 by the tie rate.
        keep = value[left] != value[right]
        keep &= (group[left] == group[right]) if within else (group[left] != group[right])
        left, right = left[keep], right[keep]

        # Sampling is with replacement, so a small cell returns the same handful of pairs
        # thousands of times. Deduplicating is not tidiness: a repeated pair carries no extra
        # information, and counting it again would both inflate its weight in this cell's
        # estimate and let `min_pairs` pass on a cell that holds nowhere near that many
        # distinct pairs — a six-row cell clears a 200-pair floor on repeats alone.
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        _, unique = np.unique(low * size + high, return_index=True)
        left, right = left[unique], right[unique]
        if len(left) < min_pairs:
            continue
        agree = (score[left] > score[right]) == (value[left] > value[right])
        out[cell] = float(agree.mean())

    return pd.Series(out, dtype="float64", name="accuracy")


def comparability(
    scores: pd.Series,
    groups: pd.Series,
    truth: pd.Series,
    cells: pd.Series,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    min_pairs: int = MIN_PAIRS,
) -> pd.DataFrame:
    """Within-group against cross-group ordering accuracy, and the paired difference.

    **The paired difference is the answer**, not the two levels. Cells differ in how hard they
    are, so comparing an unpaired within-group mean against an unpaired cross-group mean mixes
    that variation into the contrast; pairing on the cell removes it. An interval straddling zero
    says the score means the same thing in both settings, which is the claim.

    Returns:
        Three rows — ``within``, ``cross`` and ``difference`` — each with ``cells``, ``estimate``
        and its bootstrap interval. ``difference`` covers only cells that reported both.
    """
    inside = pair_accuracy(
        scores, groups, truth, cells, within=True, draws=draws, seed=seed, min_pairs=min_pairs
    )
    across = pair_accuracy(
        scores, groups, truth, cells, within=False, draws=draws, seed=seed, min_pairs=min_pairs
    )
    shared = inside.index.intersection(across.index)

    rows = []
    for name, values in (
        ("within", inside),
        ("cross", across),
        ("difference", across.loc[shared] - inside.loc[shared]),
    ):
        mean, low, high = bootstrap_ci(values)
        rows.append(
            {"pairs": name, "cells": len(values), "estimate": mean, "low": low, "high": high}
        )
    return pd.DataFrame(rows).set_index("pairs")

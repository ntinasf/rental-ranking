"""Is a ranking score comparable across query groups, or only within one?

The question decides what the system can serve. If a score means the same thing in two searches,
a **broad search** — one naming a city, a room type and a party size but no neighbourhood — is
just one search over a wider candidate set, and the served model can rank it. If it does not,
there is no way to merge and the broad search cannot be served at all.

Measured out-of-fold over universes where two query groups differ **only** by neighbourhood, it
is comparable: pairs drawn across groups are ordered as accurately as pairs inside one, and the
paired difference is indistinguishable from zero.

Calibration and comparability are different properties. A LambdaMART score is genuinely
uncalibrated — not a probability, arbitrary scale — but ordering two listings needs only a
monotone common function, and the booster is one: it takes no query-group input, so no per-group
parameter can differ. The lambdarank objective leaves cross-group ordering *unconstrained* by
training, which is not the same as destroying it.

Three decisions make the measurement mean what it says:

* **Pairwise accuracy, not NDCG**, which normalises by the candidate set and is therefore a
  different quantity at every grouping. A pair either orders correctly or it does not.
* **Every cell is fold-pure.** Out-of-fold scores come from four fold models with four scales,
  and a broad universe spans folds, so the evaluation cell is ``(universe, fold)``.
* **Cells are the resampling unit, not pairs**, which are heavily dependent inside a cell.

**Scope is a hard boundary.** Grades are quartiles within ``city x room_type``, so pooling them
across grading partitions compares quantiles computed on different populations. What is measured
here is comparability **across neighbourhoods** and nothing stronger; the cross-capacity-tier
pairs on their own resolve nothing.

Pure transforms — no I/O and no ``main()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rental_ranking.evaluate.metrics import bootstrap_ci

#: What "correctly ordered" is judged against: the training target, matching
#: ``fairness.DEFAULT_TRUTH``. Measuring against the finer ``blocked_fraction_90`` would score the
#: model for a coarsening it was never asked to undo.
DEFAULT_TRUTH = "grade"

#: Pairs sampled per cell before filtering to discordant truth. Sampling rather than enumerating,
#: because a large cell holds millions of pairs and the estimate settles well below that.
DEFAULT_DRAWS = 60_000

#: Discordant pairs a cell needs before it reports an accuracy. Below this the cell's estimate is
#: noise that the bootstrap would carry at the same weight as a well-populated one.
MIN_PAIRS = 200


def evaluation_cells(universe: pd.Series, fold: pd.Series) -> pd.Series:
    """The fold-pure unit a cross-group comparison must be computed inside.

    Concatenated out-of-fold scores come from four different fitted models, and a broad universe
    spans folds. Keying the cell on both puts one score scale inside every candidate set.

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
        seed: Sampling seed; the estimate is stable across seeds.
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

        # Sampling is with replacement, so a small cell returns the same handful of pairs many
        # times over. Deduplicating is not tidiness: a repeated pair carries no extra information,
        # and counting it again would inflate its weight and let `min_pairs` pass on a cell with
        # nowhere near that many distinct pairs — a six-row cell clears a 200-pair floor on
        # repeats alone.
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

    **The paired difference is the answer**, not the two levels: cells differ in how hard they
    are, and pairing on the cell removes that variation from the contrast. An interval straddling
    zero says the score means the same thing in both settings.

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

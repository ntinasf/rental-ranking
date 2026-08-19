"""LambdaMART: the design matrix, the group array, the parameters, and the fit.

Pure functions, no I/O and no ``main()`` — ``train/train.py`` is the orchestrator, exactly as
``features/build.py`` is to ``features/``. Everything here can be called on a slice of the
feature table without knowing where the slice came from, which is what lets the same code fit a
cross-validation fold and the final refit.

**The feature list comes from ``assemble.feature_columns``, never from ``table.columns``.** The
difference is one line and it is the difference between training on the features and training
on the answer: ``grade`` is the target and ``blocked_fraction_90`` is the demand proxy the grade
was cut from.

**Categoricals are passed natively.** Five columns — ``city``, ``room_type``, ``building_type``,
``license_status``, ``host_is_local`` — reach LightGBM as pandas ``category`` dtype, so it splits
on subsets of levels rather than on an arbitrary integer order. No one-hot, no label encoding.
Two of the five are conditioners (``city`` and ``room_type`` are constant inside a query group
and can separate no pair), and they are passed anyway: a conditioner cannot discriminate but it
can still tell the model which population it is ranking in.

**NaN is passed through untouched.** LightGBM routes missing values down a learned default
branch, which is strictly more informative than any fill — and ``price`` is the project's sole
imputed column for a reason that does not generalise (its missingness tracks the label). The
nullable extension dtypes are cast to ``float64`` first, because ``Int64``/``boolean`` carry
``pd.NA`` rather than ``np.nan``; the cast preserves missingness and changes nothing else.

**The gain vector is stated, not inherited.** Grades run 0-4 and LightGBM's default
``label_gain`` is ``2**i - 1`` over a long index, so the default happens to be right — but a
truncated vector silently flattens grade 4 into grade 3, which no metric reports.
:func:`check_label_gain` asserts the observed maximum grade is addressable instead of trusting
that.

**Its NDCG is not our NDCG, and the difference is exactly the degenerate groups.** LightGBM
scores a group whose grades are all equal as 1.0; ``evaluate.metrics`` returns NaN and counts it.
Measured on dev fold 1 (79 groups, 2 degenerate): LightGBM reports 0.7013, ``ndcg_at_k`` reports
0.6935, and ``(0.6935 * 77 + 2) / 79 = 0.7013`` reconciles them exactly. Use LightGBM's number
for early stopping — it is monotone in the same thing — and ``evaluate/`` for every number that
gets reported.
"""

from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from rental_ranking.features.assemble import feature_columns
from rental_ranking.features.groups import group_sizes

#: Columns handed to LightGBM as native categoricals. ``host_is_local`` is the fifth and least
#: obvious — it is a three-level ``local``/``foreign``/``unknown`` category, not a boolean, and
#: ``unknown`` is a real level rather than a missing value.
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "city",
    "room_type",
    "building_type",
    "license_status",
    "host_is_local",
)

#: Highest grade the label produces — scheme E puts the 0.0 atom in grade 0 and quartiles the
#: rest into 1-4.
MAX_GRADE = 4

#: Exponential gain, matching ``evaluate.metrics`` so a training log and a report are the same
#: quantity rather than merely similar ones.
LABEL_GAIN: list[int] = [2**grade - 1 for grade in range(MAX_GRADE + 1)]

#: The project's headline cut-off, and what early stopping watches.
EVAL_AT: list[int] = [10]

#: Deliberately boring. The roadmap's starting point, recorded so a later sweep has something to
#: have moved *from*. ``n_estimators`` is a ceiling, not a choice — early stopping picks the
#: number, and the cross-validation median is what the final refit uses.
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "n_estimators": 2_000,
    "label_gain": LABEL_GAIN,
    "verbose": -1,
}

#: Parameters that put randomness into a fit, with the values at which they contribute none.
#: **With all of them at their defaults LightGBM is deterministic and ``random_state`` changes
#: nothing** — measured 2026-08-18, three seeds produced bit-identical predictions. That matters
#: because "five seeds, spread 0.000" reads as a stable model when it actually means there was no
#: randomness to average over. The variance that exists is over *data*, and the group bootstrap
#: in ``evaluate/report.py`` is what measures it.
DETERMINISTIC_DEFAULTS: dict[str, object] = {
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "extra_trees": False,
}


def is_stochastic(params: dict[str, object]) -> bool:
    """Whether ``random_state`` can change this fit at all.

    Row subsampling only bites when ``subsample_freq`` is above zero, so ``subsample`` alone is
    not enough; column subsampling and ``extra_trees`` bite on their own.
    """
    settings = {**DEFAULT_PARAMS, **params}
    bagging = (
        float(settings.get("subsample", 1.0)) < 1.0 and int(settings.get("subsample_freq", 0)) > 0
    )
    return (
        bagging
        or float(settings.get("colsample_bytree", 1.0)) < 1.0
        or float(settings.get("colsample_bynode", 1.0)) < 1.0
        or bool(settings.get("extra_trees", False))
    )


#: Rounds without a validation improvement before the fit stops. Generous, because with 393
#: queries the validation NDCG is noisy and a tight patience stops on noise.
EARLY_STOPPING_ROUNDS = 100


def check_label_gain(grades: pd.Series, label_gain: Sequence[float] = LABEL_GAIN) -> None:
    """Assert every observed grade is addressable by ``label_gain``.

    A gain vector shorter than the grade range does not raise inside LightGBM in any way that
    reaches a report — the top grade is simply folded into the last entry, so grade 4 is trained
    as if it were grade 3 and every metric still prints.

    Raises:
        ValueError: If a grade is negative, non-integral, or beyond the vector.
    """
    values = grades.dropna().to_numpy()
    if values.size == 0:
        raise ValueError("no grades to train against")
    if not np.array_equal(values, values.astype("int64")):
        raise ValueError("grades must be integral: label_gain is indexed by the grade")

    lowest, highest = int(values.min()), int(values.max())
    if lowest < 0:
        raise ValueError(f"grade {lowest} is negative and cannot index label_gain")
    if highest >= len(label_gain):
        raise ValueError(
            f"grade {highest} is beyond label_gain of length {len(label_gain)}, so it would be "
            f"trained as grade {len(label_gain) - 1}: the top relevance class would be flattened "
            "into the one below it and no metric would report the difference"
        )


def design_matrix(table: pd.DataFrame, features: Sequence[str] | None = None) -> pd.DataFrame:
    """The model inputs, with nullable dtypes cast and categoricals left categorical.

    Args:
        table: A feature table or any row slice of one.
        features: Column list. Defaults to :func:`~rental_ranking.features.assemble.feature_columns`,
            which is the only source that cannot accidentally include the target.

    Returns:
        A frame of the feature columns. ``Int64``/``boolean``/``bool`` become ``float64`` so
        ``pd.NA`` becomes ``np.nan``; ``category`` columns are untouched so LightGBM splits them
        natively; everything else passes through, NaN included.
    """
    columns = list(features) if features is not None else feature_columns(table)
    matrix = table[columns].copy()
    for column in columns:
        if str(matrix[column].dtype) in ("Int64", "boolean", "bool"):
            matrix[column] = matrix[column].astype("float64")
    return matrix


def training_groups(table: pd.DataFrame, groups: pd.Series | None = None) -> np.ndarray:
    """LightGBM's positional group array for ``table``, re-checked at train time.

    **BUILD_GUIDE gotcha #4, asserted a second time on purpose.** ``features/assemble.py`` checks
    contiguity when the table is written, but a training run slices, filters and re-orders it
    afterwards, and LightGBM reads the array positionally without ever seeing a group id. A
    frame that lost its sort trains on queries stitched from two different searches and reports
    a perfectly plausible number. ``group_sizes`` owns the rule and raises on both the sum and
    the contiguity; this is the call site the roadmap asks for.
    """
    return group_sizes(table["query_group"] if groups is None else groups)


def fit(
    train: pd.DataFrame,
    validation: pd.DataFrame | None = None,
    params: dict[str, object] | None = None,
    seed: int = 0,
    n_estimators: int | None = None,
    features: Sequence[str] | None = None,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> lgb.LGBMRanker:
    """Fit a LambdaMART ranker on ``train``, early-stopping on ``validation`` if given.

    Args:
        train: Rows to fit on, sorted by ``query_group``.
        validation: Rows to early-stop on. ``None`` fits the full ``n_estimators`` — which is
            what the final refit does, at the iteration count cross-validation chose.
        params: LightGBM parameters. Defaults to :data:`DEFAULT_PARAMS`.
        seed: Passed as ``random_state``. **Report the mean over several seeds and the spread,
            never the best one** — with 393 queries the seed moves the score materially.
        n_estimators: Overrides the parameter of the same name. The refit passes the
            cross-validated iteration count here.
        features: Column list, defaulting to ``feature_columns``.
        early_stopping_rounds: Patience, ignored when ``validation`` is None.

    Returns:
        The fitted ranker. ``best_iteration_`` is set when a validation set was given.

    Raises:
        ValueError: If a grade is not addressable by ``label_gain``, or if either frame is not
            sorted into contiguous query groups.
    """
    settings = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed}
    if n_estimators is not None:
        settings["n_estimators"] = n_estimators
    check_label_gain(train["grade"], settings["label_gain"])

    columns = list(features) if features is not None else feature_columns(train)
    model = lgb.LGBMRanker(**settings)
    kwargs: dict[str, object] = {}
    if validation is not None:
        check_label_gain(validation["grade"], settings["label_gain"])
        kwargs = {
            # eval_X/eval_y take one matrix or a tuple of them, never a list — a list is read
            # as the data itself. eval_group stays a list, one entry per validation set.
            "eval_X": design_matrix(validation, columns),
            "eval_y": validation["grade"],
            "eval_group": [training_groups(validation)],
            "eval_at": EVAL_AT,
            "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)],
        }

    model.fit(
        design_matrix(train, columns),
        train["grade"],
        group=training_groups(train),
        **kwargs,
    )
    return model


def predict(
    model: lgb.LGBMRanker, table: pd.DataFrame, features: Sequence[str] | None = None
) -> pd.Series:
    """Score ``table``, returned aligned to its index so it can be sliced like any other column.

    Uses ``best_iteration_`` when the model was early-stopped, so a model fitted with a
    validation set scores with the trees it actually chose rather than every tree it grew.
    """
    columns = list(features) if features is not None else feature_columns(table)
    best = getattr(model, "best_iteration_", None)
    scores = model.predict(design_matrix(table, columns), num_iteration=best or None)
    return pd.Series(scores, index=table.index, name="score")


def serving_metadata(table: pd.DataFrame, features: Sequence[str] | None = None) -> dict:
    """The feature order and category levels a served model needs to reproduce training inputs.

    **Measured 2026-08-18, after an initial guess that was wrong in two of three parts.** What a
    served LightGBM model actually does with categoricals:

    * **Category *order* does not matter.** The booster stores the training levels in
      ``pandas_categorical`` and re-maps an incoming ``category`` column *by label* at predict
      time. A request covering only ``{local, unknown}`` of ``{foreign, local, unknown}`` scores
      identically to the full frame — verified to 0.000000. The "codes shift silently" story is
      false; LightGBM already solved it.
    * **A plain string column fails loudly.** ``json.loads`` yields ``object`` dtype, and
      predicting on that raises ``ValueError: train and valid dataset categorical_feature do not
      match``. Loud is fine; it just has to be converted.
    * **An unseen level is the one silent failure.** ``room_type="Houseboat"`` predicts without
      complaint and lands **0.1083** away from the truth, because ``set_categories`` turns the
      unknown label into NaN and the model scores it as *missing* rather than as *invalid*.

    So this metadata exists for the third case and for column order (``Booster.predict`` on a
    bare frame is positional). :func:`restore_dtypes` is where it is enforced.

    Args:
        table: The training frame, whose dtypes define the contract.
        features: Column list, defaulting to ``feature_columns``.

    Returns:
        ``{"features": [...], "categories": {column: [level, ...]}}``, JSON-serialisable, to be
        written beside the booster and read back by the scoring script.
    """
    columns = list(features) if features is not None else feature_columns(table)
    categories = {
        column: [str(level) for level in table[column].cat.categories]
        for column in columns
        if str(table[column].dtype) == "category"
    }
    return {"features": columns, "categories": categories}


def restore_dtypes(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Rebuild a request frame into the shape the model can score, and reject what it cannot.

    The inverse of :func:`serving_metadata`, and the function a scoring script calls. It does
    exactly two things that matter, and both were established by measurement rather than assumed:

    1. **Casts the JSON-borne string columns back to ``category``.** Without this the request
       fails loudly (``train and valid dataset categorical_feature do not match``), so this half
       is about working at all, not about correctness.
    2. **Rejects levels the model never saw.** This is the half that matters, because it is the
       one path that is *silent*: ``set_categories`` maps an unknown label to NaN, the model
       scores it as a missing value, and the caller receives a confident number **0.1083** away
       from the truth with no indication anything was wrong.

    Missing feature *columns* are a different matter and are allowed — they become NaN, which
    LightGBM routes down a learned branch. A caller with no review scores for a brand-new
    listing is describing the world accurately, not making a mistake.

    3. **Coerces object-dtype numeric columns to float.** Added 2026-08-19 after the cold-start
       demonstration request crashed the scoring container. ``reindex`` gives an *absent* column
       ``float64`` NaN, but a column that is **present and null for every listing** comes out of
       ``json.loads`` as ``object``, and LightGBM rejects the frame with ``pandas dtypes must be
       int, float or bool`` — an unhandled exception, so a 500 rather than a message. The two
       cases are the same request semantically; only the serialiser differs. A value that is
       genuinely not a number now raises here instead, naming the column.

    Raises:
        ValueError: If a categorical column carries a level absent from training, or a numeric
            column carries something that is not a number.
    """
    rebuilt = frame.reindex(columns=metadata["features"])
    for column, levels in metadata["categories"].items():
        values = rebuilt[column].astype("object")
        unseen = set(values.dropna().astype(str)) - set(levels)
        if unseen:
            raise ValueError(
                f"column {column!r} carries level(s) {sorted(unseen)} the model never saw during "
                f"training. Known levels: {levels}. Scoring them would silently treat the value "
                "as missing rather than reject it"
            )
        rebuilt[column] = pd.Categorical(values, categories=levels)

    for column in metadata["features"]:
        if column in metadata["categories"] or rebuilt[column].dtype != object:
            continue
        numeric = pd.to_numeric(rebuilt[column], errors="coerce")
        bad = rebuilt[column].notna() & numeric.isna()
        if bad.any():
            raise ValueError(
                f"column {column!r} carries {int(bad.sum())} non-numeric value(s), e.g. "
                f"{rebuilt.loc[bad, column].iloc[0]!r}. Numeric features must arrive as numbers "
                "or null"
            )
        rebuilt[column] = numeric.astype("float64")

    return design_matrix(rebuilt, metadata["features"])


def feature_importance(
    model: lgb.LGBMRanker, features: Sequence[str] | None = None
) -> pd.DataFrame:
    """Gain and split importance, highest gain first.

    **Read it against notebook 03 §2, not on its own.** A third of the numeric features are
    conditioners — near-constant inside a query group — so they cannot separate a pair however
    high they rank, and establishment features topping the chart re-derives what Phase 1 already
    measured rather than discovering anything.
    """
    names = list(features) if features is not None else list(model.feature_name_)
    frame = pd.DataFrame(
        {
            "gain": model.booster_.feature_importance("gain"),
            "split": model.booster_.feature_importance("split"),
        },
        index=names,
    )
    frame["gain_share"] = frame["gain"] / frame["gain"].sum()
    frame.index.name = "feature"
    return frame.sort_values("gain", ascending=False)

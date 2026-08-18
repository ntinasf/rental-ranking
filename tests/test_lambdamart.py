"""Tests for rental_ranking.train.lambdamart.

Three of these guard failures that produce a *plausible number* rather than an error, which is
the only kind worth this much test code at this point in the project: a gain vector too short
for the grade range (grade 4 trained as grade 3), a frame that lost its query-group sort
(LightGBM slicing queries across search boundaries), and a deterministic parameter set reported
as a stable one.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.train import lambdamart as lm


def _table(groups: list[int], seed: int = 0, signal: bool = True) -> pd.DataFrame:
    """A feature-table-shaped frame: identifiers, both targets, and three features."""
    rng = np.random.default_rng(seed)
    rows = []
    for group, size in enumerate(groups):
        for i in range(size):
            grade = int(rng.integers(0, 5))
            rows.append(
                {
                    "id": f"{group}_{i}",
                    "query_group": group,
                    "cluster_id": group * 1000 + i,
                    "grade": grade,
                    "blocked_fraction_90": grade / 4,
                    "strong": float(grade) if signal else rng.normal(),
                    "noise": rng.normal(),
                    "city": ["athens", "crete"][group % 2],
                }
            )
    frame = pd.DataFrame(rows)
    frame["city"] = frame["city"].astype("category")
    return frame


# --- the gain vector ---------------------------------------------------------------------------


def test_label_gain_covers_the_project_grade_range() -> None:
    assert lm.LABEL_GAIN == [0, 1, 3, 7, 15]
    assert len(lm.LABEL_GAIN) == lm.MAX_GRADE + 1


def test_a_grade_beyond_the_gain_vector_raises() -> None:
    """The silent version folds the top class into the one below and reports nothing."""
    with pytest.raises(ValueError, match="flattened"):
        lm.check_label_gain(pd.Series([0, 1, 5]))


def test_a_short_gain_vector_raises_on_ordinary_grades() -> None:
    with pytest.raises(ValueError, match="beyond label_gain"):
        lm.check_label_gain(pd.Series([0, 1, 2, 3, 4]), label_gain=[0, 1, 3])


@pytest.mark.parametrize(
    ("grades", "match"),
    [([-1, 0, 1], "negative"), ([0.5, 1.5], "integral"), ([], "no grades")],
)
def test_unusable_grades_raise(grades: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        lm.check_label_gain(pd.Series(grades, dtype="float64"))


# --- the design matrix -------------------------------------------------------------------------


def test_design_matrix_excludes_identifiers_and_both_targets() -> None:
    """``blocked_fraction_90`` is the demand proxy the grade was cut from. It is not a feature."""
    matrix = lm.design_matrix(_table([12, 12]))
    for forbidden in ("id", "query_group", "cluster_id", "grade", "blocked_fraction_90"):
        assert forbidden not in matrix.columns


def test_nullable_dtypes_become_float_and_keep_their_missingness() -> None:
    frame = _table([8, 8])
    frame["flag"] = pd.array([True, False, pd.NA] * 5 + [True], dtype="boolean")
    frame["count"] = pd.array([1, pd.NA] * 8, dtype="Int64")
    matrix = lm.design_matrix(frame)

    assert matrix["flag"].dtype == "float64"
    assert matrix["count"].dtype == "float64"
    assert matrix["flag"].isna().sum() == 5
    assert matrix["count"].isna().sum() == 8


def test_categoricals_stay_categorical_so_lightgbm_splits_them_natively() -> None:
    matrix = lm.design_matrix(_table([10, 10]))
    assert str(matrix["city"].dtype) == "category"


def test_ordinary_nan_is_not_filled() -> None:
    frame = _table([10, 10])
    frame.loc[frame.index[:4], "noise"] = np.nan
    assert lm.design_matrix(frame)["noise"].isna().sum() == 4


# --- the group array ---------------------------------------------------------------------------


def test_training_groups_rejects_a_frame_that_lost_its_sort() -> None:
    """BUILD_GUIDE gotcha #4, at the second of its two call sites. The sum still checks out."""
    frame = _table([6, 6, 6])
    shuffled = frame.sample(frac=1.0, random_state=0)
    assert lm.training_groups(frame).sum() == len(frame)
    with pytest.raises(ValueError, match="not sorted by query group"):
        lm.training_groups(shuffled)


# --- fitting -----------------------------------------------------------------------------------


def test_a_fit_learns_an_informative_feature() -> None:
    frame = _table([60] * 6, signal=True)
    model = lm.fit(frame, params={"min_child_samples": 5, "n_estimators": 40})
    scores = lm.predict(model, frame)

    assert scores.index.equals(frame.index)
    assert scores.corr(frame["grade"].astype("float64")) > 0.8


def test_early_stopping_sets_best_iteration_and_predict_uses_it() -> None:
    train, valid = _table([50] * 6, seed=1), _table([50] * 4, seed=2)
    model = lm.fit(
        train, valid, params={"min_child_samples": 5, "n_estimators": 400}, early_stopping_rounds=5
    )
    assert model.best_iteration_ is not None
    assert model.best_iteration_ <= 400
    assert len(lm.predict(model, valid)) == len(valid)


def test_n_estimators_overrides_the_parameter_set() -> None:
    frame = _table([40] * 4)
    model = lm.fit(frame, params={"min_child_samples": 5}, n_estimators=17)
    assert model.booster_.num_trees() == 17


def test_a_grade_the_gain_vector_cannot_address_stops_the_fit() -> None:
    frame = _table([20, 20])
    frame.loc[frame.index[0], "grade"] = 9
    with pytest.raises(ValueError, match="flattened"):
        lm.fit(frame)


# --- determinism -------------------------------------------------------------------------------


def test_the_starting_parameters_are_deterministic() -> None:
    """Measured 2026-08-18. Five seeds give bit-identical predictions, so a reported spread of
    0.000 means there was no randomness to average over — not that the model is stable."""
    assert not lm.is_stochastic({})
    frame = _table([40] * 5)
    first = lm.predict(lm.fit(frame, seed=0, n_estimators=30), frame)
    other = lm.predict(lm.fit(frame, seed=99, n_estimators=30), frame)
    assert np.array_equal(first.to_numpy(), other.to_numpy())


@pytest.mark.parametrize(
    "params",
    [
        {"colsample_bytree": 0.8},
        {"colsample_bynode": 0.8},
        {"extra_trees": True},
        {"subsample": 0.8, "subsample_freq": 1},
    ],
)
def test_stochastic_parameter_sets_are_recognised(params: dict) -> None:
    assert lm.is_stochastic(params)


def test_row_subsampling_without_a_frequency_is_still_deterministic() -> None:
    """``subsample`` alone does nothing: LightGBM only bags when ``subsample_freq`` is above 0."""
    assert not lm.is_stochastic({"subsample": 0.5})


# --- importance --------------------------------------------------------------------------------


def test_importance_is_sorted_by_gain_and_shares_sum_to_one() -> None:
    frame = _table([50] * 5)
    importance = lm.feature_importance(
        lm.fit(frame, params={"min_child_samples": 5}, n_estimators=30)
    )

    assert importance["gain"].is_monotonic_decreasing
    assert importance["gain_share"].sum() == pytest.approx(1.0)
    assert "grade" not in importance.index

"""Tests for rental_ranking.features.amenities.

The map is a hand-built artefact over a 7,029-string vocabulary, so its failure mode is not a
crash — it is a plausible miscategorisation nobody notices. Two classes of guard therefore
matter most:

* **Rule order.** The buckets resolve first-match, and three pairs are traps: a pool table is
  not a pool, a pool view is not a pool, an outdoor kitchen is not a kitchen appliance.
  Reordering :data:`AMENITY_BUCKETS` breaks these silently and every coverage number stays
  identical.
* **Canonicalisation reaching the rules.** A keyword that is not itself in canonical form can
  never match anything, and the map would quietly lose a whole concept while still returning
  a bucket name for everything else.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.data.paths import PROCESSED_DIR
from rental_ranking.features import amenities

# --- canonicalisation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Air Conditioning", "air conditioning"),
        ("Pack ’n play/Travel crib", "pack 'n play/travel crib"),  # curly apostrophe
        ("Free washer – In unit", "free washer - in unit"),  # en dash
        ("43 inch HDTV with Netflix", "hdtv with netflix"),  # size qualifier
        ("55 inch  HDTV", "hdtv"),
        ("  Wifi  ", "wifi"),
    ],
)
def test_canonicalise_folds_the_feeds_variants(raw: str, expected: str) -> None:
    assert amenities.canonicalise(raw) == expected


def test_canonicalise_keeps_subtype_detail() -> None:
    """Subtypes are kept so a substring rule absorbs them without an entry of its own."""
    assert "coffee" in amenities.canonicalise("Coffee maker: Nespresso")


def test_every_keyword_is_itself_canonical() -> None:
    """A non-canonical keyword can never match, and the concept vanishes silently."""
    for name, keywords in amenities.AMENITY_BUCKETS:
        for keyword in keywords:
            assert amenities.canonicalise(keyword) == keyword.strip(), (
                f"{name}: {keyword!r} is not in canonical form, so it can never match"
            )


# --- the substring traps, which rule order exists to resolve -----------------------------------


@pytest.mark.parametrize(
    ("amenity", "bucket"),
    [
        ("Pool table", "entertainment"),  # a game, not a pool
        ("Pool view", "view"),  # an outlook, not a pool
        ("Beach view", "view"),
        ("Sea view", "view"),
        ("Private pool", "pool_spa"),
        ("Shared outdoor pool - available seasonally", "pool_spa"),
        ("Outdoor kitchen", "outdoor_space"),  # a villa feature, not an appliance
        ("Kitchen", "kitchen"),
    ],
)
def test_rule_order_resolves_the_substring_traps(amenity: str, bucket: str) -> None:
    assert amenities.bucket_of(amenity) == bucket


def test_a_view_is_never_evidence_of_the_thing_seen() -> None:
    """Stated as a property rather than a case list: every '<place> view' is a view."""
    for place in ("Sea", "Mountain", "Pool", "Beach", "Garden", "City skyline", "Marina"):
        assert amenities.bucket_of(f"{place} view") == "view"


# --- subtype absorption ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("amenity", "bucket"),
    [
        ("AC - split type ductless system", "air_conditioning"),
        ("Central air conditioning", "air_conditioning"),
        ("Window AC unit", "air_conditioning"),
        ("Heating - split type ductless system", "heating"),
        ("Coffee maker: espresso machine, Nespresso", "kitchen"),
        ("43 inch HDTV with Netflix", "entertainment"),
        ("Bosch refrigerator", "kitchen"),
        ("Free washer – In unit", "laundry"),
        ("Clothing storage: closet, wardrobe, and dresser", "basics"),
    ],
)
def test_variants_collapse_onto_their_base_concept(amenity: str, bucket: str) -> None:
    assert amenities.bucket_of(amenity) == bucket


def test_the_four_air_conditioning_variants_are_one_concept() -> None:
    """The point of the whole module: the feed splits AC across mutually exclusive subtypes.

    A frequency-ranked flag encoding would carry them as four separate columns at 70.6 / 13.7 /
    9.4 / 5.0 %, and none of them would answer "does this listing have air conditioning".
    """
    variants = [
        "Air conditioning",
        "AC - split type ductless system",
        "Central air conditioning",
        "Window AC unit",
    ]
    assert {amenities.bucket_of(v) for v in variants} == {"air_conditioning"}


# --- parking, and the conditional exclusion ----------------------------------------------------


@pytest.mark.parametrize(
    ("amenity", "bucket"),
    [
        ("Free parking on premises", "parking_private"),
        ("Paid parking lot on premises", "parking_private"),
        ("Free driveway parking on premises – 2 spaces", "parking_private"),
        ("Free residential garage on premises – 1 space", "parking_private"),
        ("Free street parking", "parking_street"),
        ("Paid parking off premises", "parking_street"),
        ("Paid parking lot off premises", "parking_street"),
    ],
)
def test_on_premises_and_off_premises_parking_are_different_promises(
    amenity: str, bucket: str
) -> None:
    assert amenities.bucket_of(amenity) == bucket


def test_an_amenity_available_on_request_is_not_an_amenity() -> None:
    """ "Crib - available upon request" is a host who can find one, not a listing that has one."""
    assert amenities.bucket_of("Crib") == "family"
    assert amenities.bucket_of("Crib - available upon request") is None
    assert amenities.bucket_of("Paid pack ’n play/travel crib - available upon request") is None


def test_a_paid_amenity_is_still_the_amenity() -> None:
    """Unlike the conditional qualifier: paid parking is parking, at a price."""
    assert amenities.bucket_of("Paid parking off premises") == "parking_street"


def test_an_unmatched_amenity_returns_none_rather_than_an_other_bucket() -> None:
    """A gap in a hand-built map must stay countable as a gap."""
    assert amenities.bucket_of("Trebuchet") is None


# --- the constant itself -----------------------------------------------------------------------


def test_bucket_names_track_the_map_and_are_unique() -> None:
    assert amenities.BUCKET_NAMES == tuple(name for name, _ in amenities.AMENITY_BUCKETS)
    assert len(set(amenities.BUCKET_NAMES)) == len(amenities.BUCKET_NAMES)


def test_every_bucket_is_reachable() -> None:
    """A bucket whose keywords are all shadowed by an earlier rule is dead weight.

    Each keyword is probed embedded in a phrase rather than bare, because some are written
    with a leading space on purpose — `" view"` must not match `"viewing"`.
    """
    for name, keywords in amenities.AMENITY_BUCKETS:
        assert any(
            amenities.bucket_of(f"private {keyword.strip()} area") == name for keyword in keywords
        ), f"{name}: every keyword is claimed by an earlier rule"


# --- against the real snapshots ----------------------------------------------------------------


def test_the_map_covers_the_real_vocabulary() -> None:
    """98.9 % of mentions on the current snapshots. A rotation that breaks this is worth seeing.

    Coverage of *mentions* rather than of distinct strings is the honest measure: 93 % of the
    vocabulary appears in under 0.1 % of listings, so string coverage flatters any map.
    """
    path = PROCESSED_DIR / "listings.parquet"
    if not path.exists():
        pytest.skip(f"processed layer not on disk: {path}")

    lists = pd.read_parquet(path, columns=["amenities"])["amenities"]
    mentions = pd.Series([a for row in lists for a in row]).value_counts()
    mapped = pd.Series({s: amenities.bucket_of(s) is not None for s in mentions.index})

    assert mentions[mapped].sum() / mentions.sum() > 0.97


# --- the feature block -------------------------------------------------------------------------


def _frame(*rows: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"amenities": [np.array(r, dtype=object) for r in rows]})


def test_buckets_count_within_each_concept() -> None:
    frame = _frame(["Kitchen", "Oven", "Wifi"], ["Wifi"])
    out = amenities.amenity_features(frame)

    assert out["amenity_kitchen"].tolist() == [2, 0]
    assert out["amenity_connectivity_work"].tolist() == [1, 1]
    assert out["amenity_count"].tolist() == [3, 1]


def test_every_bucket_gets_a_column_even_when_always_zero() -> None:
    """The column set must not depend on the data, or two runs produce different matrices."""
    out = amenities.amenity_features(_frame(["Wifi"]))

    assert [f"amenity_{b}" for b in amenities.BUCKET_NAMES] == [
        c for c in out.columns if c != "amenity_count"
    ]


def test_bucket_counts_never_exceed_the_total() -> None:
    """Unmapped and conditional amenities are counted in the total and in no bucket."""
    frame = _frame(["Wifi", "Trebuchet", "Crib - available upon request"])
    out = amenities.amenity_features(frame)
    buckets = [c for c in out.columns if c.startswith("amenity_") and c != "amenity_count"]

    assert out["amenity_count"].iloc[0] == 3
    assert out[buckets].sum(axis=1).iloc[0] == 1
    assert out["amenity_family"].iloc[0] == 0


def test_the_count_scheme_is_the_baseline_alone() -> None:
    assert amenities.amenity_features(_frame(["Wifi", "Oven"]), "count").columns.tolist() == [
        "amenity_count"
    ]


def test_the_flags_scheme_needs_a_pinned_vocabulary() -> None:
    with pytest.raises(ValueError, match="pinned vocabulary"):
        amenities.amenity_features(_frame(["Wifi"]), "flags")


def test_flags_mark_presence_of_the_pinned_amenities() -> None:
    out = amenities.amenity_features(_frame(["Wifi"], ["Oven"]), "flags", ["Wifi", "Oven"])

    assert out["amenity_has_wifi"].tolist() == [1, 0]
    assert out["amenity_has_oven"].tolist() == [0, 1]


def test_an_unknown_scheme_raises() -> None:
    with pytest.raises(ValueError, match="unknown amenity scheme"):
        amenities.amenity_features(_frame(["Wifi"]), "svd")


def test_features_keep_the_callers_index() -> None:
    frame = _frame(["Wifi"], ["Oven"]).set_axis([7, 9])
    assert amenities.amenity_features(frame).index.tolist() == [7, 9]


def test_fit_vocabulary_ranks_by_frequency() -> None:
    frame = _frame(["Wifi", "Oven"], ["Wifi"], ["Wifi"])
    assert amenities.fit_vocabulary(frame, k=1) == ["Wifi"]


def test_the_variance_criterion_requires_the_query_groups() -> None:
    """It measures the variation a ranker can use, which is defined only inside a group."""
    with pytest.raises(ValueError, match="needs the query groups"):
        amenities.fit_vocabulary(_frame(["Wifi"]), by="within_group_variance")


def test_an_unknown_selection_criterion_raises() -> None:
    with pytest.raises(ValueError, match="unknown selection criterion"):
        amenities.fit_vocabulary(_frame(["Wifi"]), by="mean_label")


def test_the_pinned_vocabularies_are_usable_and_distinct() -> None:
    """Pinned so the feature set cannot depend on the split; 44 of 50 overlap by measurement."""
    for vocabulary in (amenities.FREQUENCY_TOP_50, amenities.VARIANCE_TOP_50):
        assert len(vocabulary) == 50
        assert len(set(vocabulary)) == 50

    out = amenities.amenity_features(_frame(["Wifi"]), "flags", amenities.FREQUENCY_TOP_50)
    assert len([c for c in out.columns if c.startswith("amenity_has_")]) == 50
    assert out["amenity_has_wifi"].iloc[0] == 1

"""Amenity canonicalisation and the concept-bucket map.

``amenities`` arrives from the processed layer as a parsed list of strings, never null. Across the
ranked population it holds **7,029 distinct strings**, 93 % of them in fewer than 0.1 % of
listings — a vocabulary dominated by subtype and brand variants ("43 inch HDTV with Netflix",
"Bosch refrigerator"). One-hot is impossible, and a per-string dictionary is brittle: it scores
"Fast wifi - 500 Mbps" as zero because the string is not "Wifi".

So this module does two things and stops. It **canonicalises** a raw string, and it maps the
canonical form to one of a small set of **concept buckets** by ordered keyword rules. Building
features from the buckets belongs to the caller.

**Buckets rather than a weighted score.** A hand-built dictionary of per-amenity "convenience"
weights fails on both counts: **rarity is not value**, so no frequency-derived weighting can
stand in for judgement ("Body soap" at 49.5 % is rarer than "Air conditioning" at 70.6 %), and a
cardinal scale asks its author to defend fifty numbers where a partition asks only that "parking
is a category of convenience". **Assert the partition; let the model assert the weights.**

**The map is built blind to the label**, from the vocabulary and its prevalence alone, and
nothing in this file may ever be tuned against an evaluation score — a dictionary fitted by
watching NDCG is target encoding performed through the author's eyes, and no test here would
catch it.

**Rule order is load-bearing**, because the vocabulary sets traps for substring matching: "Pool
table" is a game, "Pool view" is a view, and "Outdoor kitchen" is a villa feature rather than a
kitchen appliance. The first matching rule wins, so the specific bucket is listed before the
general one. :func:`bucket_of` returns ``None`` for anything unmatched rather than inventing an
"other" bucket — an unmatched string is a gap in the map and should be visible as one.

Pure functions, no I/O and no ``main()``.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

import pandas as pd

from rental_ranking.data.validate import require_columns

#: Qualifiers that mean the amenity is **not present by default**. Dropped rather than counted:
#: "Crib - available upon request" is a host who *can* find a crib, not a listing that has one,
#: and counting the two alike overstates every family-equipped property. Conservative by choice
#: — flip this to an empty tuple to count them and the buckets widen accordingly.
#: "Paid" variants are **not** excluded: paid parking is parking, at a price.
CONDITIONAL_QUALIFIERS: tuple[str, ...] = ("available upon request",)

#: Screen sizes ("43 inch HDTV") and similar numeric qualifiers, stripped before matching so
#: every television collapses onto one concept instead of one per diagonal.
_SIZE_QUALIFIER = re.compile(r"\b\d+\s*inch\b")

_WHITESPACE = re.compile(r"\s+")

#: The concept buckets, **in resolution order** — the first rule whose keyword appears in the
#: canonical string wins. Traps first: `pool table` before `pool`, ` view` before every place
#: noun, `outdoor kitchen` before `kitchen`. Keywords are matched as plain substrings of the
#: canonical form, so a subtype variant ("coffee maker: nespresso") lands with its base concept
#: without needing its own entry.
AMENITY_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # --- traps, resolved before the general rules they would otherwise be caught by ---
    (
        "entertainment",
        (
            "pool table",
            "tv",
            "netflix",
            "game console",
            "board games",
            "life size games",
            "books and reading",
            "sound system",
            "piano",
            "movie theater",
            "record player",
            "arcade",
            "ping pong",
            "mini golf",
        ),
    ),
    # Every "<place> view" string: sea, mountain, city skyline, harbor, pool, beach, vineyard.
    # A view is an attribute of the outlook, never evidence of the thing seen.
    ("view", (" view",)),
    # --- outdoor and water, the dimensions a Greek summer market actually turns on ---
    (
        "pool_spa",
        ("pool", "hot tub", "sauna", "jacuzzi", "plunge"),
    ),
    (
        "beach_water",
        ("beach", "waterfront", "lake access", "resort access", "ski-in", "boat slip"),
    ),
    (
        "outdoor_space",
        (
            "outdoor kitchen",
            "patio",
            "balcony",
            "backyard",
            "garden",
            "courtyard",
            "terrace",
            "outdoor furniture",
            "outdoor dining",
            "outdoor shower",
            "sun lounger",
            "bbq",
            "barbecue",
            "fire pit",
            "hammock",
            "outdoor playground",
        ),
    ),
    # --- climate: split, because an August market in Greece prices these nothing alike ---
    (
        "air_conditioning",
        ("air conditioning", "ac -", "window ac", "portable fan", "ceiling fan", "portable ac"),
    ),
    ("heating", ("heating", "heater", "fireplace", "radiant", "heat pump", "thermostat")),
    # --- parking: on-premises and off-premises are different promises, so different buckets.
    # In a Greek city centre a street-parking "amenity" guarantees nothing; a spot on the
    # property does. The split is the assertion; the model decides what each is worth.
    (
        "parking_private",
        (
            "parking on premises",
            "parking lot on premises",
            "driveway parking",
            "garage",
            "carport",
            "ev charger",
        ),
    ),
    ("parking_street", ("street parking", "parking off premises", "parking lot off premises")),
    # --- interior blocks ---
    (
        "kitchen",
        (
            "kitchen",
            "kitchenette",
            "oven",
            "stove",
            "microwave",
            "dishwasher",
            "refrigerator",
            "mini fridge",
            "freezer",
            "cooking basics",
            "dishes and silverware",
            "coffee",
            "kettle",
            "toaster",
            "blender",
            "wine glasses",
            "dining table",
            "baking sheet",
            "rice maker",
            "bread maker",
            "trash compactor",
            "children's dinnerware",
        ),
    ),
    ("laundry", ("washer", "dryer", "drying rack", "iron")),
    (
        "family",
        (
            "crib",
            "high chair",
            "pack 'n play",
            "travel crib",
            "changing table",
            "baby",
            "children",
            "outlet covers",
            "table corner guards",
            "window guards",
            "stair gates",
            "playroom",
        ),
    ),
    ("connectivity_work", ("wifi", "ethernet", "workspace", "internet")),
    (
        "safety",
        (
            "smoke alarm",
            "carbon monoxide",
            "fire extinguisher",
            "first aid",
            "lock on bedroom door",
            "safe",
            "security camera",
            "noise decibel",
            "lockbox",
            "keypad",
            "smart lock",
            "gated community",
        ),
    ),
    (
        "access",
        (
            "self check-in",
            "host greets you",
            "luggage dropoff",
            "elevator",
            "private entrance",
            "single level home",
            "building staff",
            "doorman",
            "step-free",
            "wide ",
            "accessible",
        ),
    ),
    ("fitness", ("gym", "exercise equipment", "bikes", "kayak")),
    ("services", ("cleaning available", "housekeeping", "breakfast", "laundromat", "concierge")),
    (
        "basics",
        (
            "shampoo",
            "conditioner",
            "body soap",
            "shower gel",
            "essentials",
            "hangers",
            "hair dryer",
            "bed linens",
            "extra pillows",
            "hot water",
            "clothing storage",
            "room-darkening",
            "cleaning products",
            "bathtub",
            "bidet",
            "mosquito net",
            "towels",
            "toilet paper",
            "shower",
            "closet",
            "dresser",
            "wardrobe",
            "sofa bed",
            "mattress",
            "living room",
            "clothes drying",
        ),
    ),
    # Policies rather than conveniences, bucketed apart so a caller can drop them wholesale.
    ("policy", ("pets allowed", "smoking allowed", "long term stays", "events allowed")),
)

#: Bucket names in map order, for building a stable column set.
BUCKET_NAMES: tuple[str, ...] = tuple(name for name, _ in AMENITY_BUCKETS)

#: The top 50 amenities by **frequency**, fitted once on the ranked population and pinned so the
#: feature set does not depend on the split.
FREQUENCY_TOP_50: tuple[str, ...] = (
    "Wifi",
    "Hair dryer",
    "Kitchen",
    "Hot water",
    "Hangers",
    "Iron",
    "Bed linens",
    "Dishes and silverware",
    "Cooking basics",
    "Essentials",
    "Refrigerator",
    "Air conditioning",
    "Fire extinguisher",
    "Shampoo",
    "Hot water kettle",
    "First aid kit",
    "TV",
    "Shower gel",
    "Wine glasses",
    "Smoke alarm",
    "Long term stays allowed",
    "Self check-in",
    "Drying rack for clothing",
    "Dining table",
    "Extra pillows and blankets",
    "Outdoor furniture",
    "Heating",
    "Body soap",
    "Oven",
    "Room-darkening shades",
    "Coffee",
    "Cleaning products",
    "Dedicated workspace",
    "Private entrance",
    "Freezer",
    "Washer",
    "Private patio or balcony",
    "Microwave",
    "Lockbox",
    "Free street parking",
    "Baking sheet",
    "Coffee maker",
    "Toaster",
    "Outdoor dining area",
    "Free parking on premises",
    "Luggage dropoff allowed",
    "Stove",
    "Elevator",
    "Dishwasher",
    "Crib",
)

#: The top 50 by **mean within-query-group variance**, same population. The criterion matched to
#: the objective: a pairwise ranker can only learn from variation inside a group.
#:
#: The two lists overlap in 44 of 50, so at k = 50 the choice is close to a tie. It bites hardest
#: at small k, where frequency spends the whole budget on commodity.
VARIANCE_TOP_50: tuple[str, ...] = (
    "Heating",
    "Body soap",
    "Smoke alarm",
    "Extra pillows and blankets",
    "Long term stays allowed",
    "Free street parking",
    "Room-darkening shades",
    "Dedicated workspace",
    "Shower gel",
    "First aid kit",
    "Outdoor furniture",
    "Wine glasses",
    "Coffee",
    "TV",
    "Coffee maker",
    "Drying rack for clothing",
    "Self check-in",
    "Dining table",
    "Cleaning products",
    "Hot water kettle",
    "Washer",
    "Private patio or balcony",
    "Oven",
    "Shampoo",
    "Fire extinguisher",
    "Freezer",
    "Luggage dropoff allowed",
    "Refrigerator",
    "Air conditioning",
    "Stove",
    "Baking sheet",
    "Lockbox",
    "Outdoor dining area",
    "Microwave",
    "Essentials",
    "Private entrance",
    "Toaster",
    "Bed linens",
    "Cooking basics",
    "Clothing storage",
    "Dishes and silverware",
    "Patio or balcony",
    "Carbon monoxide alarm",
    "Iron",
    "Exterior security cameras on property",
    "Crib",
    "Elevator",
    "Clothing storage: wardrobe",
    "Hangers",
    "Host greets you",
)


def canonicalise(amenity: str) -> str:
    """Fold a raw amenity string onto the form the bucket rules match against.

    Lowercases, normalises unicode (the feed mixes ``’`` with ``'`` and ``–`` with ``-``),
    strips screen-size qualifiers, and collapses whitespace. Subtype and brand detail is
    deliberately **kept** — ``"coffee maker: nespresso"`` still contains ``"coffee"``, so the
    substring rules absorb the variant without an entry of its own, and nothing is lost that a
    later, finer map might want.

    Args:
        amenity: One raw amenity string as it appears in the processed layer.

    Returns:
        The canonical form: lowercase, ASCII-normalised punctuation, single-spaced.
    """
    text = unicodedata.normalize("NFKC", amenity).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = _SIZE_QUALIFIER.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def bucket_of(amenity: str) -> str | None:
    """Map one amenity to its concept bucket, or ``None`` if no rule matches.

    The first matching rule in :data:`AMENITY_BUCKETS` wins, which is what resolves the
    substring traps — see the module docstring. Strings carrying a
    :data:`CONDITIONAL_QUALIFIERS` phrase return ``None``: the amenity is not present by
    default, so it belongs in no bucket rather than in a weaker version of one.

    ``None`` is returned rather than an ``"other"`` bucket on purpose. An unmatched string is a
    gap in a hand-built map, and it should be countable as a gap rather than absorbed into a
    category that then means "everything I did not think of".

    Args:
        amenity: A raw amenity string; canonicalisation happens here.

    Returns:
        The bucket name, or ``None`` when nothing matches or the amenity is conditional.
    """
    text = canonicalise(amenity)
    if any(qualifier in text for qualifier in CONDITIONAL_QUALIFIERS):
        return None
    for name, keywords in AMENITY_BUCKETS:
        if any(keyword in text for keyword in keywords):
            return name
    return None


def fit_vocabulary(
    listings: pd.DataFrame,
    k: int = 50,
    by: str = "frequency",
    groups: pd.Series | None = None,
) -> list[str]:
    """Select the ``k`` amenities a flag encoding should carry, **to be pinned as a constant**.

    Fitting at training time would make the feature set depend on the split, so the intended use
    is: run this once on the ranked population, paste the result into a constant, and pass that
    constant to :func:`amenity_features` afterwards. Both criteria read only the amenity lists, so
    the reason to pin is reproducibility rather than leakage.

    Two criteria, because the obvious one is wrong for a ranker:

    * ``"frequency"`` — the most common amenities. Its weakness is measurable: the top of the
      vocabulary is commodity (Wifi 92.5 %, Hair dryer 89.9 %), so a flag block spends its budget
      on amenities almost every listing has.
    * ``"within_group_variance"`` — mean variance of the flag *inside* a query group, which is
      the only variation LambdaMART can learn from: a pair is only ever compared against
      another member of its own group. An amenity every listing in a group shares contributes
      no pairwise preference however common or rare it is globally.

    Args:
        listings: Frame carrying ``amenities``.
        k: How many amenities to keep.
        by: ``"frequency"`` or ``"within_group_variance"``.
        groups: Query-group ids aligned to ``listings``. Required for the variance criterion.

    Returns:
        ``k`` raw amenity strings, best first.

    Raises:
        KeyError: If ``amenities`` is absent.
        ValueError: If ``by`` is unknown, or the variance criterion is asked for without
            ``groups``.
    """
    require_columns(listings, ("amenities",), "listings")
    counts = Counter()
    for row in listings["amenities"]:
        counts.update(set(row))

    if by == "frequency":
        return [name for name, _ in Counter(counts).most_common(k)]
    if by != "within_group_variance":
        raise ValueError(f"unknown selection criterion {by!r}")
    if groups is None:
        raise ValueError(
            "the within-group-variance criterion needs the query groups: it measures the "
            "variation a ranker can actually use, which is defined only inside a group"
        )

    # Only amenities with some global spread can have within-group spread, and scoring all
    # 7,029 costs a groupby each. The 500 most common is a generous superset of any k.
    sets = listings["amenities"].map(set)
    scores: dict[str, float] = {}
    for name, _ in Counter(counts).most_common(500):
        flag = sets.map(lambda row, n=name: n in row).astype("float64")
        scores[name] = float(flag.groupby(groups).var().mean())
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:k]]


def amenity_features(
    listings: pd.DataFrame,
    scheme: str = "buckets",
    vocabulary: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the amenity feature block under one of the candidate encodings.

    The scheme is a **parameter rather than a fork in the code**, so a comparison between
    encodings varies exactly one thing and cannot drift between variants. The candidates:

    * ``"buckets"`` — one **count** per concept bucket, the default. Counts rather than flags,
      because 8 of the 19 buckets are near-universal on presence, where a flag is dead weight
      while the count inside the bucket still varies. A count also subsumes a flag: a tree
      recovers presence by splitting at zero.
    * ``"count"`` — ``amenity_count`` alone. The honest baseline, and not a naive one: it
      correlates with the label at the same order as the review signals. If the bucket block
      cannot beat it, the buckets were encoding listing verbosity.
    * ``"flags"`` — one binary per amenity in ``vocabulary``, which must be pinned; see
      :func:`fit_vocabulary`.

    ``amenity_count`` is emitted by every scheme. It is orthogonal to the rest and it is the
    control: bucket counts sum to *less* than it, because unmapped and conditional amenities
    are excluded by design.

    Args:
        listings: Frame carrying ``amenities`` as lists.
        scheme: ``"buckets"``, ``"count"`` or ``"flags"``.
        vocabulary: Amenity strings for ``"flags"``; ignored otherwise.

    Returns:
        A frame aligned to ``listings``, all columns prefixed ``amenity_``.

    Raises:
        KeyError: If ``amenities`` is absent.
        ValueError: If ``scheme`` is unknown, or ``"flags"`` is asked for with no vocabulary.
    """
    require_columns(listings, ("amenities",), "listings")
    if scheme not in {"buckets", "count", "flags"}:
        raise ValueError(f"unknown amenity scheme {scheme!r}")

    rows = listings["amenities"]
    features = pd.DataFrame({"amenity_count": rows.str.len().astype("int64")}, index=listings.index)
    if scheme == "count":
        return features

    if scheme == "flags":
        if not vocabulary:
            raise ValueError(
                "scheme 'flags' needs a pinned vocabulary — fit it once with fit_vocabulary "
                "and pass the constant, so the feature set does not depend on the split"
            )
        sets = rows.map(set)
        for name in vocabulary:
            features[f"amenity_has_{_slug(name)}"] = sets.map(lambda row, n=name: n in row).astype(
                "int64"
            )
        return features

    # Buckets. One pass per listing, resolving each amenity through the cached map: the
    # vocabulary is 7,029 strings against 1.74 M mentions, so caching the lookup matters.
    resolved: dict[str, str | None] = {}
    tallies = []
    for row in rows:
        tally: Counter = Counter()
        for item in row:
            if item not in resolved:
                resolved[item] = bucket_of(item)
            bucket = resolved[item]
            if bucket is not None:
                tally[bucket] += 1
        tallies.append(tally)

    counts = (
        pd.DataFrame(tallies, index=listings.index)
        .reindex(columns=list(BUCKET_NAMES))
        .fillna(0)
        .astype("int64")
        .add_prefix("amenity_")
    )
    return pd.concat([features, counts], axis=1)


def _slug(amenity: str) -> str:
    """A column-safe name for a flag, stable across snapshots for the same amenity string."""
    text = canonicalise(amenity)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")

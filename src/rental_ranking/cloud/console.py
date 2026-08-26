"""A local console for the ranking endpoint: edit a real listing, watch where it lands.

Serves a small page with the real category levels in dropdowns, seeded from a **real sealed-fold
listing** inside its **real candidate set**, and renders the response against the **held-out
grades**. Change a field, press re-rank, see the listing move. Starting from a real listing is
what keeps the answer checkable: an invented row carries no grade, so its ranking cannot be
judged.

``city`` and ``room_type`` are displayed but not editable — they are constant inside a query
group by construction, so changing either would build a shared room competing inside an
entire-homes search, a row the model was never asked to score.

The server is a proxy rather than a nicety. A browser cannot call the endpoint directly (managed
online endpoints send no CORS headers), and putting the auth key in the page would leak a live
credential; the key stays in this process. Standard library only — ``http.server`` and
``urllib``.

Run it::

    uv run python -m rental_ranking.cloud.console --local   # no endpoint, no cost
    uv run python -m rental_ranking.cloud.console           # against AML_ENDPOINT_URI
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from rental_ranking.cloud import demo
from rental_ranking.cloud.score import ID_FIELD as ID_COLUMN

#: Columns the console shows but never lets you change: they are the query-group key, constant
#: inside a group, so editing one produces a candidate the group was never asked to rank.
FIXED_CONTEXT: tuple[str, ...] = ("city", "room_type")

#: The editable form, grouped the way a person reads a listing rather than the way the model
#: consumes it. ``step`` is the input's granularity; ``None`` means the field takes an integer.
#: This is a *subset* of the 61 features on purpose — every feature would be a wall of boxes, and
#: the ones left out (twenty amenity buckets, four host-portfolio counts) move a rank far less
#: than the ones kept.
FORM_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("property", ("accommodates", "bedrooms", "beds", "bathrooms", "amenity_count")),
    ("price", ("price", "minimum_nights", "price_vs_nbhd")),
    (
        "reviews",
        (
            "number_of_reviews",
            "number_of_reviews_ltm",
            "reviews_per_month",
            "reviews_same_season_ly",
            "rating_shrunk",
            "review_scores_accuracy",
            "review_scores_cleanliness",
            "review_scores_checkin",
            "review_scores_communication",
            "review_scores_location",
            "review_scores_value",
            "days_since_last_review",
        ),
    ),
    (
        "host",
        (
            "host_is_superhost",
            "host_is_local",
            "host_tenure_months",
            "license_status",
            "building_type",
            "calculated_host_listings_count",
        ),
    ),
    ("location", ("km_to_nearest_anchor", "km_to_neighbourhood_centroid", "density_1km")),
    ("age", ("listing_age_days", "has_reviews")),
)

#: One-click edits. Each is a whole story rather than a field: the first is the counterfactual
#: from the transcript, the second asks what the model does with a listing that has never been
#: reviewed, the third moves only price so the response can be attributed to it.
PRESETS: dict[str, dict[str, Any]] = {
    "strip review history": demo.COUNTERFACTUAL_BLANKS,
    "make it brand new": demo.COLD_START_BLANKS,
    "double the price": {},  # resolved per listing in :func:`preset_edits`
}

DEFAULT_PORT = 8000


def preset_columns() -> set[str]:
    """Every column any preset touches. Must be a subset of the editable form — see the test.

    A preset writes into the form and the form is what gets sent, so a preset naming a field the
    form does not render would silently drop that edit.
    """
    named = {column for edits in PRESETS.values() for column in edits}
    return named | {"price"}  # "double the price" resolves per listing, so it is not in the map


def editable_columns() -> set[str]:
    """Every column the form renders."""
    return {column for _, columns in FORM_SECTIONS for column in columns}


# --- the form -------------------------------------------------------------------------------


def _kind(value: object, column: str, categories: Mapping[str, Sequence[str]]) -> str:
    if column in categories:
        return "choice"
    if isinstance(value, bool):
        return "boolean"
    return "number"


def field_spec(listing: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[dict]:
    """The form definition for one listing: every editable field, its kind and its current value.

    Fields the served model does not carry are dropped rather than rendered dead, so the page can
    never offer an input the endpoint will ignore.
    """
    categories = metadata["categories"]
    served = set(metadata["features"])
    spec: list[dict] = []
    for section, columns in FORM_SECTIONS:
        fields = [
            {
                "name": column,
                "section": section,
                "kind": _kind(listing.get(column), column, categories),
                "value": listing.get(column),
                "choices": list(categories.get(column, ())),
            }
            for column in columns
            if column in served and column not in FIXED_CONTEXT
        ]
        spec.extend(fields)
    return spec


def preset_edits(name: str, listing: Mapping[str, Any]) -> dict[str, Any]:
    """The edit map for a named preset, resolved against the listing it applies to.

    Raises:
        KeyError: If the preset is unknown; the page only offers names from :data:`PRESETS`.
    """
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
    if name == "double the price":
        price = listing.get("price")
        return {"price": None if price is None else float(price) * 2}
    return dict(PRESETS[name])


def coerce_edits(edits: Mapping[str, Any], spec: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Turn the form's strings back into the types the endpoint expects.

    An empty box becomes ``null`` — the caller saying the listing has no value for that field,
    which is a real answer rather than an error.

    Raises:
        ValueError: If a numeric field holds something that is not a number, or a choice field
            holds a level the model has never seen. Both would otherwise reach the scorer.
    """
    kinds = {field["name"]: field for field in spec}
    out: dict[str, Any] = {}
    for name, raw in edits.items():
        field = kinds.get(name)
        if field is None:
            raise ValueError(f"{name!r} is not an editable field")
        if raw is None or raw == "":
            out[name] = None
            continue
        if field["kind"] == "boolean":
            out[name] = raw in (True, "true", "on", "1", 1)
        elif field["kind"] == "choice":
            if raw not in field["choices"]:
                raise ValueError(f"{name!r}: {raw!r} is not one of {field['choices']}")
            out[name] = raw
        else:
            try:
                out[name] = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name!r}: {raw!r} is not a number") from error
    return out


# --- the answer the page renders ---------------------------------------------------------------


#: Rows the table draws. The largest real query group is 2,088 listings; a page that renders all
#: of them is unreadable and slow, and NDCG@10 only ever looks at the first ten anyway. The edited
#: listing is always included, wherever it landed.
DISPLAY_LIMIT = 100


def ranking_view(
    response: Mapping[str, Any],
    truth: pd.DataFrame,
    edited: str,
    before: Mapping[str, Any] | None = None,
    k: int = demo.DEFAULT_K,
    limit: int = DISPLAY_LIMIT,
) -> dict:
    """The response, the truth beside it, and the movement — everything the page draws.

    Every group reaching here is from the sealed fold (:func:`demo.group_listings` refuses the
    rest), so every metric in the payload is out-of-sample.

    Args:
        response: The ranking under the current edits.
        truth: Held-out grades and readable attributes, indexed by id.
        edited: The listing being edited, so the page can mark its row.
        before: The unedited ranking, if there is one, to report the movement against.
        k: Metric cut-off.
        limit: Rows to return. The edited listing is always among them.

    Returns:
        A JSON-serialisable dict: ``rows``, ``n_rows``, ``edited``, ``k``, ``metrics`` and
        ``moved``.
    """
    table = demo.explain(response, truth, k=k).reset_index()
    shown = table.head(limit)
    if edited not in set(shown["id"]):
        shown = pd.concat([shown, table[table["id"] == edited]])

    quality = demo.query_quality(response, truth, k=k)
    moved = None
    if before is not None:
        prior = demo.query_quality(before, truth, k=k)
        moved = {
            "rank_before": demo.rank_of(before, edited),
            "rank_after": demo.rank_of(response, edited),
            "ndcg_before": float(prior["endpoint"]),
            "ndcg_after": float(quality["endpoint"]),
        }
    return {
        "rows": json.loads(shown.to_json(orient="records")),
        "n_rows": len(table),
        "edited": edited,
        "k": k,
        "metrics": {name: float(value) for name, value in quality.items()},
        "moved": moved,
    }


# --- the page ---------------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>rental-ranker console</title>
<style>
:root {
  --bg:#fbfaf8; --panel:#fff; --line:#e2ded7; --ink:#1c1a17; --dim:#6b665e;
  --accent:#7a4b2a; --warn:#8a6d1f; --up:#1f7a4d; --down:#b03a2e;
  --g0:#e6e3de; --g1:#d9e2ea; --g2:#bdd3e0; --g3:#8fc0a9; --g4:#5f9e78;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#171614; --panel:#1f1e1b; --line:#34322d; --ink:#ece8e1; --dim:#98928a;
          --accent:#d19a6a; --warn:#d8b45e; --up:#63c191; --down:#e07a6b;
          --g0:#2c2a26; --g1:#2f3b45; --g2:#3a5566; --g3:#3f6b52; --g4:#4f8a67; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; }
h1 { margin:0 0 12px; font-size:15px; letter-spacing:.02em; }
h1 span { color:var(--dim); font-weight:400; }
header { padding:16px 22px 14px; border-bottom:1px solid var(--line); background:var(--panel); }

/* the search bar — four dropdowns and a button, in the order a guest fills them */
.search { display:flex; gap:0; align-items:stretch; flex-wrap:wrap;
  border:1px solid var(--line); border-radius:999px; background:var(--bg); padding:4px;
  max-width:1000px; }
.cell { display:flex; flex-direction:column; justify-content:center; padding:4px 16px;
  border-right:1px solid var(--line); min-width:0; flex:1 1 160px; }
.cell:nth-last-child(2) { border-right:0; }
.cell b { font-size:10px; text-transform:uppercase; letter-spacing:.09em; color:var(--dim); }
.cell select { border:0; background:transparent; color:var(--ink); padding:1px 0;
  font:13px ui-sans-serif,system-ui,sans-serif; width:100%; }
.search button { border-radius:999px; margin:0 2px; padding:0 22px; }
.quick { margin-top:10px; display:flex; gap:6px; align-items:center; flex-wrap:wrap;
  font-size:12px; color:var(--dim); }

.wrap { display:grid; grid-template-columns:minmax(290px,340px) 1fr; align-items:start; }
@media (max-width:980px) { .wrap { grid-template-columns:1fr; } }
aside { border-right:1px solid var(--line); padding:0 20px 22px; }
main { padding:16px 22px 40px; overflow-x:auto; }
.sticky { position:sticky; top:0; background:var(--bg); padding:16px 0 10px; z-index:2;
  border-bottom:1px solid var(--line); }

.banner { border:1px solid var(--line); background:var(--panel); border-radius:6px;
  padding:9px 12px; margin-bottom:14px; font-size:12.5px; }
.banner.pooled { border-color:var(--warn); }
.coverage { color:var(--dim); font-size:11.5px; margin-top:9px; max-width:96ch; }
.coverage b { color:var(--ink); font-weight:600; }
.banner small { display:block; color:var(--dim); margin-top:3px; }
.banner b { font-weight:600; }

fieldset { border:0; border-top:1px solid var(--line); margin:0; padding:11px 0 3px; }
legend { color:var(--accent); font-size:10px; text-transform:uppercase; letter-spacing:.09em; padding-right:8px; }
label { display:grid; grid-template-columns:1fr 108px; gap:8px; align-items:center; margin-bottom:5px; }
label span { color:var(--dim); font-size:12px; overflow:hidden; text-overflow:ellipsis; }
input,select { background:var(--panel); color:var(--ink); border:1px solid var(--line);
  border-radius:4px; padding:4px 6px; width:100%;
  font:12px ui-monospace,SFMono-Regular,Menlo,monospace; }
input[type=checkbox] { width:auto; justify-self:start; }
label.edited input, label.edited select { border-color:var(--accent); }
button { background:var(--accent); color:#fff; border:0; border-radius:5px; padding:7px 14px;
  font:600 13px ui-sans-serif,system-ui,sans-serif; cursor:pointer; }
button.ghost { background:transparent; color:var(--accent); border:1px solid var(--line); font-weight:400; }
button:disabled { opacity:.45; cursor:default; }
.row { display:flex; gap:6px; flex-wrap:wrap; margin:8px 0 0; }

table { border-collapse:collapse; width:100%;
  font:12px ui-monospace,SFMono-Regular,Menlo,monospace; }
th { text-align:right; color:var(--dim); font-weight:500; padding:5px 9px;
  border-bottom:1px solid var(--line); white-space:nowrap; }
td { text-align:right; padding:4px 9px; border-bottom:1px solid var(--line); white-space:nowrap; }
td.name, th.name { text-align:left; max-width:240px; overflow:hidden; text-overflow:ellipsis; }
td.id, th.id { text-align:left; }
tr.me td { background:color-mix(in srgb, var(--accent) 15%, transparent); font-weight:600; }
tr.hit td.name { color:var(--accent); }
tr.cut td { border-bottom:2px solid var(--accent); }
.chip { display:inline-block; min-width:20px; padding:1px 6px; border-radius:9px; text-align:center; }
.g0{background:var(--g0)}.g1{background:var(--g1)}.g2{background:var(--g2)}.g3{background:var(--g3)}.g4{background:var(--g4)}
.cards { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:8px 12px; min-width:118px; }
.card b { display:block; font:600 17px ui-monospace,monospace; }
.card small { color:var(--dim); font-size:11px; }
.move { font-size:13px; margin:0 0 12px; min-height:20px; }
.up { color:var(--up); } .down { color:var(--down); }
.note { color:var(--dim); font-size:12px; max-width:74ch; margin-top:16px; }
.note a { color:var(--accent); }
.err { color:var(--down); font:12px ui-monospace,monospace; white-space:pre-wrap; margin-top:8px; }
</style>

<header>
  <h1>rental-ranker <span>&mdash; search a real candidate set, score it live, read it against the grades</span></h1>
  <div class="search">
    <div class="cell"><b>city</b><select id="s_city"></select></div>
    <div class="cell"><b>neighbourhood</b><select id="s_nbhd"></select></div>
    <div class="cell"><b>room type</b><select id="s_room"></select></div>
    <div class="cell"><b>guests</b><select id="s_guests"></select></div>
    <button id="go">search</button>
  </div>
  <div class="quick" id="quick">worked examples:</div>
  <p class="coverage" id="coverage"></p>
</header>

<div class="wrap">
<aside>
  <div class="sticky">
    <select id="listing" title="listing being edited"></select>
    <div class="row">
      <button type="button" id="rank">re-rank</button>
      <button type="button" class="ghost" id="reset">reset</button>
    </div>
    <div class="row" id="presets"></div>
    <div class="err" id="err"></div>
  </div>
  <form id="form"></form>
</aside>

<main>
  <div class="banner" id="banner"></div>
  <div class="cards" id="cards"></div>
  <p class="move" id="move"></p>
  <div id="table"></div>
  <p class="note" id="note"></p>
  <p class="note">Listing data from <a href="https://insideairbnb.com/">Inside Airbnb</a>, licensed
    <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Listing ids are hashed and
    host identifiers are stripped before anything leaves the data layer. The occupancy target is a
    <b>demand proxy</b> built from forward-looking calendar availability — it is not booking history.</p>
</main>
</div>

<script>
let OPT = null, STATE = null, HITS = new Set();

const $ = id => document.getElementById(id);
const uniq = xs => [...new Set(xs)].sort((a, b) => String(a).localeCompare(String(b), "el"));
const fmt = (v, d) => v === null || v === undefined ? ""
  : (typeof v === "number" ? v.toFixed(d === undefined ? 3 : d) : v);
const fill = (sel, values, keep) => {
  sel.innerHTML = values.map(v => `<option>${v}</option>`).join("");
  if (keep && values.includes(keep)) sel.value = keep;
};

/* --- the search bar: each dropdown is filtered by the ones to its left --------------------- */

function refineSearch(from) {
  const rows = OPT.index;
  const city = $("s_city").value;
  if (from <= 0) fill($("s_nbhd"), uniq(rows.filter(r => r.city === city).map(r => r.neighbourhood)), $("s_nbhd").value);
  const nbhd = $("s_nbhd").value;
  if (from <= 1) fill($("s_room"), uniq(rows.filter(r => r.city === city && r.neighbourhood === nbhd).map(r => r.room_type)), $("s_room").value);
  const room = $("s_room").value;
  const tiers = uniq(rows.filter(r => r.city === city && r.neighbourhood === nbhd && r.room_type === room).map(r => r.tier));
  const guests = tiers.flatMap(t => OPT.guests[t] || []).sort((a, b) => a - b);
  fill($("s_guests"), guests, Number($("s_guests").value));
}

async function search() {
  const q = new URLSearchParams({
    city: $("s_city").value, neighbourhood: $("s_nbhd").value,
    room_type: $("s_room").value, guests: $("s_guests").value,
  });
  await load("/api/search?" + q);
}

/* --- loading a candidate set ---------------------------------------------------------------- */

async function load(url) {
  $("err").textContent = "";
  const out = await (await fetch(url)).json();
  if (out.error) { $("err").textContent = out.error; return; }
  STATE = out;
  // Only worth marking when the search is narrower than the group it landed in. If every
  // listing matched, highlighting every listing says nothing.
  HITS = new Set(out.search && out.search.matching < out.n ? out.search.matching_ids : []);
  fill($("listing"), []);
  $("listing").innerHTML = STATE.listings.map((l, i) =>
    `<option value="${l.id}">#${i + 1} · grade ${l.grade} · ${l.name || l.id}</option>`).join("");
  $("listing").value = STATE.edited;
  drawBanner(); drawForm(); draw(STATE.baseline_view);
}

function drawBanner() {
  const s = STATE.search, b = $("banner");
  b.className = "banner" + (s && s.pooled ? " pooled" : "");
  const k = STATE.key;
  const where = k.neighbourhood || `${k.neighbourhoods} neighbourhoods`;
  const tier = k.capacity_tier ? `${k.capacity_tier} guests` : `${k.tiers} capacity tiers`;
  const head = s
    ? `<b>${s.city} · ${s.neighbourhood} · ${s.room_type} · ${s.guests} guests</b> ` +
      `&rarr; query group <b>${STATE.query_group}</b>, ${STATE.n} listings`
    : `<b>${k.city} · ${where} · ${k.room_type} · ${tier}</b> ` +
      `&rarr; query group <b>${STATE.query_group}</b>, ${STATE.n} listings`;
  const tail = !s ? "the query-group key is city × neighbourhood × room type × capacity tier &mdash; " +
      "the same four things a guest picks, which is why searching for them selects a candidate set"
    : s.pooled
      ? `only ${s.matching} sealed listing(s) match that exact search, so the group was pooled at a ` +
        `coarser key and spans <b>${s.neighbourhoods} neighbourhoods</b>. The competition is city-wide, ` +
        `not local &mdash; that is the fallback in features/groups.py, not a bug`
      : `all ${STATE.n} listings share that neighbourhood and capacity tier, so the group formed on ` +
        `the full key. Matching listings are highlighted in the table`;
  b.innerHTML = head + `<small>${tail}</small>`;
}

/* --- the editor ------------------------------------------------------------------------------ */

function drawForm() {
  const bySection = {};
  STATE.fields.forEach(f => (bySection[f.section] ||= []).push(f));
  $("form").innerHTML = Object.entries(bySection).map(([s, fs]) =>
    `<fieldset><legend>${s}</legend>` + fs.map(f => {
      const id = "f_" + f.name;
      if (f.kind === "choice")
        return `<label><span title="${f.name}">${f.name}</span><select id="${id}" data-name="${f.name}">` +
          f.choices.map(c => `<option${c === f.value ? " selected" : ""}>${c}</option>`).join("") +
          `</select></label>`;
      if (f.kind === "boolean")
        return `<label><span title="${f.name}">${f.name}</span>` +
          `<input type="checkbox" id="${id}" data-name="${f.name}"${f.value ? " checked" : ""}></label>`;
      // Shown rounded so it fits the box; submitted at full precision unless actually edited, so
      // an untouched field cannot perturb the baseline it is being compared against.
      const exact = f.value === null ? "" : String(f.value);
      const shown = f.value === null ? "" : String(Math.round(f.value * 1e4) / 1e4);
      return `<label><span title="${f.name}">${f.name}</span>` +
        `<input id="${id}" data-name="${f.name}" data-exact="${exact}" data-shown="${shown}" value="${shown}"></label>`;
    }).join("") + `</fieldset>`).join("");
}

function readForm() {
  const edits = {};
  document.querySelectorAll("#form [data-name]").forEach(el => {
    if (el.type === "checkbox") { edits[el.dataset.name] = el.checked; return; }
    const untouched = el.dataset.shown !== undefined && el.value === el.dataset.shown;
    edits[el.dataset.name] = untouched ? el.dataset.exact : el.value;
  });
  return edits;
}

function applyEdits(edits) {
  Object.entries(edits).forEach(([name, value]) => {
    const el = document.querySelector(`#form [data-name="${name}"]`);
    if (!el) return;
    if (el.type === "checkbox") el.checked = !!value;
    else { el.value = value === null ? "" : value; el.dataset.shown = el.value; el.dataset.exact = el.value; }
    el.closest("label").classList.add("edited");
  });
}

async function rank(edits) {
  $("err").textContent = "";
  const body = {query_group: STATE.query_group, listing_id: $("listing").value, edits};
  const out = await (await fetch("/api/rank", {method: "POST", body: JSON.stringify(body)})).json();
  if (out.error) { $("err").textContent = out.error; return; }
  draw(out);
}

/* Reset reloads this group at this listing: original values, no movement line, no error.
   Three bugs lived here. It used to snap back to the top-ranked listing rather than the one you
   were editing; it left a refused edit's error on screen; and it was disabled until a re-rank
   *succeeded*, so after a rejected edit — the moment you most want it — the button did nothing.
   It is now always live, because "put this back" is always a valid thing to ask. */
async function reset() {
  await load(`/api/group?query_group=${STATE.query_group}&listing=${$("listing").value}`);
}

/* --- results ---------------------------------------------------------------------------------- */

const COLS = [["rank", 0], ["id", null], ["name", null], ["score", 4], ["grade", null],
              ["blocked_fraction_90", 3], ["number_of_reviews", 0], ["rating_shrunk", 3],
              ["price", 2], ["listing_age_days", 0]];

function draw(view) {
  const m = view.metrics;
  $("cards").innerHTML = [
    ["endpoint", m.endpoint], ["reviews", m.baseline_reviews],
    ["price+rating", m.baseline_price_rating], ["random floor", m.random],
  ].map(([n, v]) => `<div class="card"><b>${v.toFixed(4)}</b><small>${n}</small></div>`).join("");

  const mv = view.moved;
  $("move").innerHTML = !mv ? "" :
    `<code>${view.edited}</code> &nbsp; rank <b>${mv.rank_before} → ${mv.rank_after}</b> of ${view.n_rows} ` +
    `<span class="${mv.rank_after < mv.rank_before ? "up" : mv.rank_after > mv.rank_before ? "down" : ""}">` +
    `${mv.rank_after === mv.rank_before ? "(unmoved)" : mv.rank_after < mv.rank_before ? "▲" : "▼"}</span>` +
    ` &nbsp;·&nbsp; NDCG@${view.k} ${mv.ndcg_before.toFixed(4)} → ${mv.ndcg_after.toFixed(4)}`;

  $("table").innerHTML =
    `<table><thead><tr>` + COLS.map(([c]) =>
      `<th class="${c === "name" ? "name" : c === "id" ? "id" : ""}">${c}</th>`).join("") +
    `</tr></thead><tbody>` +
    view.rows.map(r => {
      const cls = [r.id === view.edited ? "me" : "", HITS.has(r.id) ? "hit" : "",
                   r.rank === view.k ? "cut" : ""].filter(Boolean).join(" ");
      return `<tr class="${cls}">` + COLS.map(([c, d]) =>
        c === "grade" ? `<td><span class="chip g${r.grade}">${r.grade}</span></td>`
        : c === "name" ? `<td class="name" title="${(r.name || "").replace(/"/g, "&quot;")}">${r.name || ""}</td>`
        : c === "id" ? `<td class="id">${r.id}</td>`
        : `<td>${fmt(r[c], d)}</td>`).join("") + `</tr>`;
    }).join("") + `</tbody></table>`;

  const truncated = view.n_rows > view.rows.length
    ? `Showing the top ${view.rows.length} of <b>${view.n_rows}</b> listings (plus the one you are ` +
      `editing, if it fell below). ` : "";
  $("note").innerHTML = truncated +
    `Grades are the truth and are never sent to the scorer. The rule under the cut line is ` +
    `NDCG@${view.k}: only the top ${view.k} count. A single query is an anecdote &mdash; the ` +
    `estimate is the sealed fold, 0.7530 [0.7148, 0.7903] over 72 groups, against 0.6429 for ` +
    `price+rating and a 0.5519 floor.`;
}

/* --- wiring ------------------------------------------------------------------------------------ */

$("s_city").onchange = () => refineSearch(0);
$("s_nbhd").onchange = () => refineSearch(1);
$("s_room").onchange = () => refineSearch(2);
$("go").onclick = search;
$("rank").onclick = () => rank(readForm());
$("reset").onclick = reset;
$("listing").onchange = e =>
  load(`/api/group?query_group=${STATE.query_group}&listing=${e.target.value}`);

(async () => {
  OPT = await (await fetch("/api/options")).json();
  const cv = OPT.coverage;
  const worst = Object.entries(cv.biggest_hidden)
    .map(([city, h]) => `${h.neighbourhood} (${city}, ${h.listings})`).join(", ");
  $("coverage").innerHTML =
    `The picker offers only <b>held-out</b> listings, so every score below is out-of-sample. ` +
    `That leaves <b>${cv.hidden} of ${cv.neighbourhoods} neighbourhoods</b> unsearchable ` +
    `&mdash; ${(cv.hidden_share * 100).toFixed(1)} % of listings, including the largest in each ` +
    `city: ${worst}. The split moves whole connected components and a large neighbourhood is a ` +
    `large component, so it lands in training entire. Showing those anyway would mean scoring ` +
    `groups the model fitted, where the ordering is a memory rather than a prediction.`;

  fill($("s_city"), uniq(OPT.index.map(r => r.city)));
  refineSearch(0);
  $("quick").innerHTML += OPT.quick_picks.map(q =>
    `<button type="button" class="ghost" data-group="${q.query_group}" title="${q.note}">${q.name}</button>`).join("");
  $("presets").innerHTML = OPT.presets.map(p =>
    `<button type="button" class="ghost" data-preset="${p}">${p}</button>`).join("");

  document.querySelectorAll("[data-group]").forEach(b =>
    b.onclick = () => load(`/api/group?query_group=${b.dataset.group}`));
  document.querySelectorAll("[data-preset]").forEach(b => b.onclick = async () => {
    const q = new URLSearchParams({name: b.dataset.preset,
      query_group: STATE.query_group, listing: $("listing").value});
    const out = await (await fetch("/api/preset?" + q)).json();
    if (out.error) { $("err").textContent = out.error; return; }
    applyEdits(out.edits);
    rank(readForm());
  });

  load(`/api/group?query_group=${OPT.quick_picks[0].query_group}`);
})();
</script>
"""


# --- the server -------------------------------------------------------------------------------


class _Session:
    """Per-query-group state, built once and reused.

    Resolving a group recomputes the fold assignment, which runs connected components over 44,684
    rows — per click that would make the console feel slow when the model is not involved.
    """

    def __init__(self, features: Sequence[str], metadata: Mapping[str, Any], send) -> None:
        self._features = list(features)
        self._metadata = metadata
        self._send = send
        self._cache: dict[int, dict] = {}

    def group(self, query_group: int) -> dict:
        if query_group not in self._cache:
            listings = demo.group_listings(query_group)
            payload = demo.build_payload(listings, self._features)
            baseline = self._send(payload)
            if "error" in baseline:
                raise ValueError(f"group {query_group}: {baseline['error']}")
            self._cache[query_group] = {
                "listings": listings,
                "payload": payload,
                "truth": demo.truth_frame(listings),
                "baseline": baseline,
            }
        return self._cache[query_group]

    def _listing(self, query_group: int, listing_id: str) -> dict:
        rows = self.group(query_group)["payload"]["listings"]
        hit = next((r for r in rows if r[ID_COLUMN] == listing_id), None)
        if hit is None:
            raise KeyError(f"{listing_id!r} is not in query group {query_group}")
        return hit

    def state(self, query_group: int, listing_id: str | None = None) -> dict:
        session = self.group(query_group)
        chosen = listing_id or session["baseline"]["ranked"][0][ID_COLUMN]
        truth = session["truth"]
        return {
            "query_group": query_group,
            "key": demo.group_key(session["listings"]),
            "n": len(session["payload"]["listings"]),
            "edited": chosen,
            "fields": field_spec(self._listing(query_group, chosen), self._metadata),
            "listings": [
                {
                    "id": row[ID_COLUMN],
                    "grade": int(truth.loc[row[ID_COLUMN], "grade"]),
                    "name": truth.loc[row[ID_COLUMN], "name"],
                }
                for row in session["baseline"]["ranked"][:DISPLAY_LIMIT]
            ],
            "baseline_view": ranking_view(session["baseline"], truth, chosen),
        }

    def search(self, city: str, neighbourhood: str, room_type: str, guests: int) -> dict:
        """Resolve one search to a candidate set, and say how wide that set really is."""
        hit = demo.resolve_search(city, neighbourhood, room_type, guests)
        session = self.group(hit["query_group"])
        matching = session["listings"]
        matching = matching[
            (matching["neighbourhood_cleansed"] == neighbourhood)
            & (matching["room_type"] == room_type)
            & (matching["capacity_tier"] == hit["capacity_tier"])
        ]
        state = self.state(hit["query_group"])
        state["search"] = {
            **hit,
            "city": city,
            "neighbourhood": neighbourhood,
            "room_type": room_type,
            "guests": guests,
            "matching_ids": matching[ID_COLUMN].tolist(),
        }
        return state

    def rank(self, query_group: int, listing_id: str, edits: Mapping[str, Any]) -> dict:
        session = self.group(query_group)
        clean = coerce_edits(
            edits, field_spec(self._listing(query_group, listing_id), self._metadata)
        )
        response = self._send(demo.perturb(session["payload"], listing_id, clean))
        if "error" in response:
            return {"error": response["error"]}
        return ranking_view(response, session["truth"], listing_id, before=session["baseline"])

    def preset(self, name: str, query_group: int, listing_id: str) -> dict:
        return {"edits": preset_edits(name, self._listing(query_group, listing_id))}

    def options(self) -> dict:
        """Everything the search bar needs, plus the three worked examples."""
        index = demo.search_index()
        return {
            "index": [
                {
                    "city": row.city,
                    "neighbourhood": row.neighbourhood_cleansed,
                    "room_type": row.room_type,
                    "tier": str(row.capacity_tier),
                }
                for row in index.itertuples()
            ],
            "guests": demo.tier_guest_choices(),
            "coverage": demo.coverage(),
            "presets": list(PRESETS),
            "quick_picks": [
                {"name": name, "query_group": spec["query_group"], "note": spec["note"]}
                for name, spec in demo.DEMO_QUERIES.items()
            ],
        }


def _handler(session: _Session):
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import parse_qs, urlparse

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            """Silence the per-request access log; it is noise in a screenshot session."""

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: object, status: int = 200) -> None:
            self._send(json.dumps(payload).encode(), "application/json", status)

        @staticmethod
        def _message(error: Exception) -> str:
            """``str(KeyError)`` wraps the message in repr quotes; the page shows it verbatim."""
            return str(error.args[0]) if isinstance(error, KeyError) and error.args else str(error)

        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming
            route = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(route.query).items()}
            try:
                if route.path in ("/", "/index.html"):
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif route.path == "/api/options":
                    self._json(session.options())
                elif route.path == "/api/group":
                    self._json(
                        session.state(int(params["query_group"]), params.get("listing") or None)
                    )
                elif route.path == "/api/search":
                    self._json(
                        session.search(
                            params["city"],
                            params["neighbourhood"],
                            params["room_type"],
                            int(params["guests"]),
                        )
                    )
                elif route.path == "/api/preset":
                    self._json(
                        session.preset(
                            params["name"], int(params["query_group"]), params["listing"]
                        )
                    )
                else:
                    self._json({"error": f"no route {route.path}"}, 404)
            except (KeyError, ValueError) as error:
                self._json({"error": self._message(error)}, 400)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                self._json(
                    session.rank(int(body["query_group"]), body["listing_id"], body["edits"])
                )
            except (KeyError, ValueError) as error:
                self._json({"error": self._message(error)}, 400)

    return Handler


def main(argv: Sequence[str] | None = None) -> None:
    """Serve the console. ``--local`` scores in-process; otherwise it proxies to the endpoint."""
    import argparse
    import webbrowser
    from http.server import ThreadingHTTPServer

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind. Loopback by default so the console is not exposed to the "
        "network; the container image overrides it to 0.0.0.0 because a container's own "
        "namespace is the boundary there",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="score in this process rather than calling the endpoint — no Azure, no cost",
    )
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)

    metadata = demo._serving_metadata()
    if args.local:
        send, source = demo.score_locally, "local scoring script (no endpoint)"
    else:
        uri, key = demo.endpoint_address()
        source = uri

        def send(payload: Mapping[str, Any]) -> dict:
            return demo.invoke(uri, key, payload)

    session = _Session(metadata["features"], metadata, send)
    address = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"  # noqa: S104
    print(f"scoring against: {source}\nconsole:         {address}\nCtrl-C to stop")

    server = ThreadingHTTPServer((args.host, args.port), _handler(session))
    if not args.no_open:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

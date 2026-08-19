"""A local console for the ranking endpoint: edit a real listing, watch where it lands.

The screenshots this project needs have to answer "does the model rank?", and a JSON body cannot
answer it. This module serves a small page with the real category levels in dropdowns, seeded
from a **real sealed-fold listing** inside its **real candidate set**, and renders the response
against the **held-out grades**. Change a field, press re-rank, see the listing move.

**Two design constraints, both about honesty rather than convenience.**

*It edits a real listing rather than composing one.* An invented row has no grade, so the ranking
it produces cannot be checked — the most it can show is that the service responds. Starting from
a listing that carries a true grade keeps the answer falsifiable, and keeps the other 22
candidates and their grades intact around it.

*The query-group key is displayed, not editable.* ``city`` and ``room_type`` are constant inside a
query group by construction (docs/decisions_log.md, 2026-08-17: anything in the key is a
conditioner, not a discriminator). A dropdown that changed either would build a shared room
competing inside an entire-homes search — a row the model was never asked to score and whose
answer means nothing.

**The server is a proxy, not a nicety.** A browser cannot call the endpoint directly: managed
online endpoints send no CORS headers, so the preflight fails, and putting the auth key in a page
would leak a live credential to anything the browser loads. The key stays in this process.

Standard library only — ``http.server`` and ``urllib``. A demonstration console is not a reason
to add a web framework to a project whose only other server is somebody else's container.

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

    The invariant matters because a preset writes into the *form* and the form is what gets sent.
    A preset naming a field the form does not render would silently drop that field: the page
    would show one edit and the endpoint would receive another, and the rank move would be
    attributed to the wrong change.
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
        KeyError: If the preset is unknown. The page only offers names from :data:`PRESETS`, so
            an unknown one means a hand-made request, and guessing at it would be worse.
    """
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
    if name == "double the price":
        price = listing.get("price")
        return {"price": None if price is None else float(price) * 2}
    return dict(PRESETS[name])


def coerce_edits(edits: Mapping[str, Any], spec: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Turn the form's strings back into the types the endpoint expects.

    An empty box becomes ``null``, which is a real answer — the caller is saying the listing has
    no value for that field — and not an error. A box that cannot be read as a number raises here
    rather than travelling to the endpoint to fail there.

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


def ranking_view(
    response: Mapping[str, Any],
    truth: pd.DataFrame,
    edited: str,
    before: Mapping[str, Any] | None = None,
    k: int = demo.DEFAULT_K,
) -> dict:
    """The response, the truth beside it, and the movement — everything the page draws.

    Args:
        response: The ranking under the current edits.
        truth: Held-out grades and readable attributes, indexed by id.
        edited: The listing being edited, so the page can mark its row.
        before: The unedited ranking, if there is one, to report the rank and NDCG move against.
        k: Metric cut-off.

    Returns:
        A JSON-serialisable dict: ``rows``, ``metrics``, ``edited``, and ``moved`` (``null`` when
        there is nothing to compare against).
    """
    table = demo.explain(response, truth, k=k).reset_index()
    quality = demo.query_quality(response, truth, k=k)

    rows = json.loads(table.to_json(orient="records"))
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
        "rows": rows,
        "metrics": {name: float(value) for name, value in quality.items()},
        "edited": edited,
        "k": k,
        "moved": moved,
    }


# --- the page ---------------------------------------------------------------------------------

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>rental-ranker console</title>
<style>
:root {
  --bg: #fbfaf8; --panel: #fff; --line: #e2ded7; --ink: #1c1a17; --dim: #6b665e;
  --accent: #7a4b2a; --up: #1f7a4d; --down: #b03a2e;
  --g0: #e6e3de; --g1: #d9e2ea; --g2: #bdd3e0; --g3: #8fc0a9; --g4: #5f9e78;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#171614; --panel:#1f1e1b; --line:#34322d; --ink:#ece8e1; --dim:#98928a;
          --accent:#d19a6a; --up:#63c191; --down:#e07a6b;
          --g0:#2c2a26; --g1:#2f3b45; --g2:#3a5566; --g3:#3f6b52; --g4:#4f8a67; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 ui-sans-serif,system-ui,sans-serif; }
header { padding:18px 22px; border-bottom:1px solid var(--line); }
h1 { margin:0; font-size:15px; letter-spacing:.02em; }
h1 span { color:var(--dim); font-weight:400; }
.wrap { display:grid; grid-template-columns:minmax(300px,360px) 1fr; gap:0; align-items:start; }
@media (max-width:900px) { .wrap { grid-template-columns:1fr; } }
aside { border-right:1px solid var(--line); padding:0 22px 18px; }
.sticky { position:sticky; top:0; background:var(--bg); padding:18px 0 10px; z-index:2;
  border-bottom:1px solid var(--line); margin-bottom:4px; }
main { padding:18px 22px; overflow-x:auto; }
.ctx { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:10px 12px; margin-bottom:16px; }
.ctx b { font-weight:600; }
.ctx small { color:var(--dim); display:block; margin-top:4px; }
fieldset { border:0; border-top:1px solid var(--line); margin:0 0 4px; padding:12px 0 4px; }
legend { color:var(--accent); font-size:11px; text-transform:uppercase; letter-spacing:.08em; padding-right:8px; }
label { display:grid; grid-template-columns:1fr 110px; gap:8px; align-items:center; margin-bottom:6px; }
label span { color:var(--dim); font-size:12px; overflow:hidden; text-overflow:ellipsis; }
input,select { background:var(--panel); color:var(--ink); border:1px solid var(--line);
  border-radius:4px; padding:4px 6px; font:12px ui-monospace,SFMono-Regular,Menlo,monospace; width:100%; }
input[type=checkbox] { width:auto; justify-self:start; }
.edited input, .edited select { border-color:var(--accent); }
button { background:var(--accent); color:#fff; border:0; border-radius:5px; padding:8px 14px;
  font:600 13px ui-sans-serif,system-ui,sans-serif; cursor:pointer; }
button.ghost { background:transparent; color:var(--accent); border:1px solid var(--line); font-weight:400; }
.row { display:flex; gap:6px; flex-wrap:wrap; margin:12px 0; }
table { border-collapse:collapse; font:12px ui-monospace,SFMono-Regular,Menlo,monospace; width:100%; }
th { text-align:right; color:var(--dim); font-weight:500; padding:5px 9px; border-bottom:1px solid var(--line); white-space:nowrap; }
td { text-align:right; padding:4px 9px; border-bottom:1px solid var(--line); white-space:nowrap; }
th:nth-child(2), td:nth-child(2) { text-align:left; }
tr.me td { background:color-mix(in srgb, var(--accent) 14%, transparent); font-weight:600; }
tr.cut td { border-bottom:2px solid var(--accent); }
.chip { display:inline-block; min-width:20px; padding:1px 6px; border-radius:9px; text-align:center; }
.g0{background:var(--g0)}.g1{background:var(--g1)}.g2{background:var(--g2)}.g3{background:var(--g3)}.g4{background:var(--g4)}
.cards { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:8px 12px; min-width:120px; }
.card b { display:block; font:600 17px ui-monospace,monospace; }
.card small { color:var(--dim); font-size:11px; }
.move { font-size:13px; margin:0 0 14px; }
.up { color:var(--up); } .down { color:var(--down); }
.note { color:var(--dim); font-size:12px; max-width:70ch; margin-top:16px; }
.err { color:var(--down); font:12px ui-monospace,monospace; white-space:pre-wrap; }
</style>

<header>
  <h1>rental-ranker <span>&mdash; sealed-fold candidate sets, live scoring, held-out grades</span></h1>
</header>

<div class="wrap">
<aside>
  <div class="sticky">
    <div class="row">
      <select id="query" style="flex:1" title="query group"></select>
    </div>
    <div class="row">
      <select id="listing" style="flex:1" title="listing being edited"></select>
    </div>
    <div class="row">
      <button type="button" id="rank">re-rank</button>
      <button type="button" class="ghost" id="reset">reset</button>
    </div>
    <div class="row" id="presets"></div>
    <div class="err" id="err"></div>
  </div>
  <div class="ctx" id="ctx"></div>
  <form id="form"></form>
</aside>

<main>
  <div class="cards" id="cards"></div>
  <p class="move" id="move"></p>
  <div id="table"></div>
  <p class="note" id="note"></p>
</main>
</div>

<script>
let STATE = null, BASE = null;

const fmt = (v, d) => v === null || v === undefined ? "" :
  (typeof v === "number" ? v.toFixed(d === undefined ? 3 : d) : v);

async function loadQuery(name) {
  const r = await fetch("/api/query?name=" + encodeURIComponent(name));
  STATE = await r.json();
  BASE = STATE.baseline;
  const sel = document.getElementById("listing");
  sel.innerHTML = STATE.listings.map(l =>
    `<option value="${l.id}">${l.id} — grade ${l.grade}</option>`).join("");
  sel.value = STATE.edited;
  drawContext(); drawForm(); draw(STATE.baseline_view);
}

function drawContext() {
  document.getElementById("ctx").innerHTML =
    `<b>group ${STATE.query_group}</b> · ${STATE.n} listings · ` +
    Object.entries(STATE.context).map(([k, v]) => `${k} <b>${v}</b>`).join(" · ") +
    `<small>fixed: these are the query-group key, constant inside the group — ` +
    `editing one would build a candidate this search never contained</small>`;
}

function drawForm() {
  const bySection = {};
  STATE.fields.forEach(f => (bySection[f.section] ||= []).push(f));
  document.getElementById("form").innerHTML = Object.entries(bySection).map(([s, fs]) =>
    `<fieldset><legend>${s}</legend>` + fs.map(f => {
      const id = "f_" + f.name;
      if (f.kind === "choice")
        return `<label><span title="${f.name}">${f.name}</span><select id="${id}" data-name="${f.name}">` +
          f.choices.map(c => `<option${c === f.value ? " selected" : ""}>${c}</option>`).join("") +
          `</select></label>`;
      if (f.kind === "boolean")
        return `<label><span title="${f.name}">${f.name}</span>` +
          `<input type="checkbox" id="${id}" data-name="${f.name}"${f.value ? " checked" : ""}></label>`;
      // Shown rounded so it fits the box; sent at full precision unless actually edited, so
      // an untouched field cannot perturb the baseline it is being compared against.
      const exact = f.value === null ? "" : String(f.value);
      const shown = f.value === null ? "" : String(Math.round(f.value * 1e4) / 1e4);
      return `<label><span title="${f.name}">${f.name}</span>` +
        `<input id="${id}" data-name="${f.name}" data-exact="${exact}" data-shown="${shown}" ` +
        `value="${shown}"></label>`;
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
  document.getElementById("err").textContent = "";
  const body = {query: STATE.name, listing_id: document.getElementById("listing").value, edits};
  const r = await fetch("/api/rank", {method: "POST", body: JSON.stringify(body)});
  const out = await r.json();
  if (out.error) { document.getElementById("err").textContent = out.error; return; }
  draw(out);
}

function draw(view) {
  const m = view.metrics;
  document.getElementById("cards").innerHTML = [
    ["endpoint", m.endpoint], ["reviews", m.baseline_reviews],
    ["price+rating", m.baseline_price_rating], ["random floor", m.random],
  ].map(([n, v]) => `<div class="card"><b>${v.toFixed(4)}</b><small>${n}</small></div>`).join("");

  const mv = view.moved;
  document.getElementById("move").innerHTML = !mv ? "" :
    `<code>${view.edited}</code> &nbsp; rank <b>${mv.rank_before} → ${mv.rank_after}</b> of ${view.rows.length}` +
    ` <span class="${mv.rank_after < mv.rank_before ? "up" : mv.rank_after > mv.rank_before ? "down" : ""}">` +
    `${mv.rank_after === mv.rank_before ? "(unmoved)" : mv.rank_after < mv.rank_before ? "▲" : "▼"}</span>` +
    ` &nbsp;·&nbsp; NDCG@${view.k} ${mv.ndcg_before.toFixed(4)} → ${mv.ndcg_after.toFixed(4)}`;

  const cols = [["rank", 0], ["id", null], ["score", 4], ["grade", null], ["blocked_fraction_90", 3],
                ["number_of_reviews", 0], ["rating_shrunk", 3], ["price", 2], ["listing_age_days", 0]];
  document.getElementById("table").innerHTML =
    `<table><thead><tr>` + cols.map(([c]) => `<th>${c}</th>`).join("") + `</tr></thead><tbody>` +
    view.rows.map(r => {
      const cls = [r.id === view.edited ? "me" : "", r.rank === view.k ? "cut" : ""].join(" ").trim();
      return `<tr class="${cls}">` + cols.map(([c, d]) =>
        c === "grade" ? `<td><span class="chip g${r.grade}">${r.grade}</span></td>`
                      : `<td>${fmt(r[c], d)}</td>`).join("") + `</tr>`;
    }).join("") + `</tbody></table>`;

  document.getElementById("note").innerHTML =
    `Grades are the <b>held-out truth</b>, never sent to the scorer. The rule under the ` +
    `cut line is NDCG@${view.k}: only the top ${view.k} count. A single query is an anecdote — ` +
    `the estimate is the sealed fold, 0.7530 [0.7148, 0.7903] over 72 groups, against 0.6429 ` +
    `for price+rating and a 0.5519 floor.`;
}

document.getElementById("rank").onclick = () => rank(readForm());
document.getElementById("reset").onclick = () => loadQuery(STATE.name);
document.getElementById("query").onchange = e => loadQuery(e.target.value);
document.getElementById("listing").onchange = async e => {
  const r = await fetch(`/api/query?name=${STATE.name}&listing=${e.target.value}`);
  STATE = await r.json(); drawForm(); draw(STATE.baseline_view);
};

(async () => {
  const meta = await (await fetch("/api/queries")).json();
  document.getElementById("query").innerHTML =
    meta.queries.map(q => `<option value="${q.name}">${q.name} — ${q.note}</option>`).join("");
  document.getElementById("presets").innerHTML = meta.presets.map(p =>
    `<button type="button" class="ghost" data-preset="${p}">${p}</button>`).join("");
  document.querySelectorAll("[data-preset]").forEach(b => b.onclick = async () => {
    const r = await fetch(`/api/preset?name=${encodeURIComponent(b.dataset.preset)}` +
      `&query=${STATE.name}&listing=${document.getElementById("listing").value}`);
    const out = await r.json();
    applyEdits(out.edits);
    rank(readForm());
  });
  loadQuery(meta.queries[0].name);
})();
</script>
"""


# --- the server -------------------------------------------------------------------------------


class _Session:
    """Per-query state, built once and reused.

    Resolving a query means reading the feature table and recomputing the fold assignment, which
    runs connected components over 44,684 rows. Doing that per keystroke would make the console
    feel like the model is slow when the model is not involved.
    """

    def __init__(self, features: Sequence[str], metadata: Mapping[str, Any], send) -> None:
        self._features = list(features)
        self._metadata = metadata
        self._send = send
        self._cache: dict[str, dict] = {}

    def query(self, name: str) -> dict:
        if name not in self._cache:
            listings, spec = demo._sealed_listings(name)
            payload = demo.build_payload(listings, self._features)
            baseline = self._send(payload)
            if "error" in baseline:
                raise ValueError(f"{name}: {baseline['error']}")
            self._cache[name] = {
                "name": name,
                "spec": spec,
                "listings": listings,
                "payload": payload,
                "truth": demo.truth_frame(listings),
                "baseline": baseline,
            }
        return self._cache[name]

    def rows(self, name: str) -> list[dict]:
        """Every listing in the query, in the unedited ranking's order — the picker's contents."""
        session = self.query(name)
        truth = session["truth"]
        return [
            {"id": row[ID_COLUMN], "grade": int(truth.loc[row[ID_COLUMN], "grade"])}
            for row in session["baseline"]["ranked"]
        ]

    def state(self, name: str, listing_id: str | None = None) -> dict:
        session = self.query(name)
        chosen = listing_id or session["baseline"]["ranked"][0][ID_COLUMN]
        listing = next(r for r in session["payload"]["listings"] if r[ID_COLUMN] == chosen)
        frame = session["listings"].set_index(ID_COLUMN)
        return {
            "name": name,
            "query_group": session["spec"]["query_group"],
            "note": session["spec"]["note"],
            "n": len(session["payload"]["listings"]),
            "context": {column: str(frame.loc[chosen, column]) for column in FIXED_CONTEXT},
            "edited": chosen,
            "fields": field_spec(listing, self._metadata),
            "listings": self.rows(name),
            "baseline": session["baseline"],
            "baseline_view": ranking_view(session["baseline"], session["truth"], chosen),
        }

    def rank(self, name: str, listing_id: str, edits: Mapping[str, Any]) -> dict:
        session = self.query(name)
        listing = next(r for r in session["payload"]["listings"] if r[ID_COLUMN] == listing_id)
        clean = coerce_edits(edits, field_spec(listing, self._metadata))
        response = self._send(demo.perturb(session["payload"], listing_id, clean))
        if "error" in response:
            return {"error": response["error"]}
        return ranking_view(response, session["truth"], listing_id, before=session["baseline"])

    def preset(self, name: str, query: str, listing_id: str) -> dict:
        session = self.query(query)
        listing = next(r for r in session["payload"]["listings"] if r[ID_COLUMN] == listing_id)
        return {"edits": preset_edits(name, listing)}


def _handler(session: _Session):
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import parse_qs, urlparse

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # noqa: A002 — silence the per-request access log
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: object, status: int = 200) -> None:
            self._send(json.dumps(payload).encode(), "application/json", status)

        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming
            route = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(route.query).items()}
            try:
                if route.path in ("/", "/index.html"):
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif route.path == "/api/queries":
                    self._json(
                        {
                            "queries": [
                                {"name": name, "note": spec["note"]}
                                for name, spec in demo.DEMO_QUERIES.items()
                            ],
                            "presets": list(PRESETS),
                        }
                    )
                elif route.path == "/api/query":
                    self._json(session.state(params["name"], params.get("listing")))
                elif route.path == "/api/preset":
                    self._json(session.preset(params["name"], params["query"], params["listing"]))
                else:
                    self._json({"error": f"no route {route.path}"}, 404)
            except (KeyError, ValueError) as error:
                self._json({"error": str(error)}, 400)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                self._json(session.rank(body["query"], body["listing_id"], body["edits"]))
            except (KeyError, ValueError) as error:
                self._json({"error": str(error)}, 400)

    return Handler


def main(argv: Sequence[str] | None = None) -> None:
    """Serve the console. ``--local`` scores in-process; otherwise it proxies to the endpoint."""
    import argparse
    import webbrowser
    from http.server import ThreadingHTTPServer

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
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
    address = f"http://127.0.0.1:{args.port}/"
    print(f"scoring against: {source}\nconsole:         {address}\nCtrl-C to stop")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), _handler(session))
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

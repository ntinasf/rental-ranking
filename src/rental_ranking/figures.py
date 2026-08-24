"""Build the figures ``docs/report.md`` embeds. The sole writer of ``docs/figures/``.

The report makes claims with numbers in them, and a figure that disagrees with the prose is
worse than no figure at all. So nothing here is drawn from a literal: the three charts are
rendered from the same code paths that produced the numbers in the first place, and everything
else is lifted straight out of the committed notebooks rather than redrawn.

Two provenance rules, which are the whole reason this module exists:

**Notebook plots are extracted, never re-run.** Each of the four notebooks carries its figures
as base64 PNGs in its committed outputs. Copying those bytes out guarantees the report shows
exactly the figure the notebook shows — no kernel, no seed, no matplotlib version in between.

**The three new charts are computed, never typed in.** The sealed comparison comes from
:func:`train.train.run`, the cold-start reach is recomputed from the out-of-fold scores that
same call returns, and the narrowing ladder is read from the CSV
``evaluate.exposure`` writes. A number in the report can therefore be wrong, but it cannot
silently disagree with the chart beside it.

Run:
    uv run python -m rental_ranking.figures
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from rental_ranking import eda
from rental_ranking.data import paths
from rental_ranking.train import baseline as bl
from rental_ranking.train import split
from rental_ranking.train import train as trainer

#: Which embedded notebook output goes to which report figure. A cell can emit several images —
#: notebook 01's leak cell renders the plot plain, then captioned, then displays the figure
#: object again — so the output index is part of the address, not an afterthought.
NOTEBOOK_FIGURES: dict[str, tuple[str, int, int]] = {
    "leak_price_quote_date.png": ("01_data_inventory.ipynb", 28, 2),
    "query_group_sizes.png": ("01_data_inventory.ipynb", 39, 0),
    "target_atoms.png": ("02_label_validation.ipynb", 13, 0),
    "atom_cohorts.png": ("02_label_validation.ipynb", 15, 0),
    "review_staircase.png": ("02_label_validation.ipynb", 31, 0),
    "amenity_count.png": ("03_feature_analysis.ipynb", 24, 0),
    "learning_curves.png": ("04_evaluation.ipynb", 17, 0),
}

#: Hand-drawn diagrams that live under ``docs/`` as self-contained HTML pages: one inline
#: ``<svg>``, no scripts, no external assets. The page stays the editable original — it is what
#: you open to change the drawing — and ``docs/figures/`` gets the two renderings a Markdown
#: document can actually embed. Name -> the page it is lifted from.
HTML_DIAGRAMS: dict[str, str] = {
    "ab_traffic_split": "ab_traffic_split.html",
}

#: Raster width for a diagram, in pixels. The pages are authored around 1240 CSS px, so 1.5x is
#: sharp on a retina screen without doubling the committed bytes.
DIAGRAM_WIDTH = 1860

#: The cut-off the whole report is written against: NDCG@10, a first screen of ten.
FIRST_SCREEN = 10


def _png_addresses(cells: list) -> list[tuple[int, int]]:
    """Every ``(cell, output)`` in a notebook that holds a PNG.

    A cell index is a brittle address: inserting one Markdown cell above a plot shifts every
    figure below it. Nothing can stop that, but the failure can at least carry the answer, so a
    drifted address is a one-line fix rather than a hunt through the JSON.
    """
    return [
        (cell_index, output_index)
        for cell_index, cell in enumerate(cells)
        for output_index, output in enumerate(cell.get("outputs", []))
        if "image/png" in output.get("data", {})
    ]


def extract_notebook_figures(
    notebooks_dir: Path, out_dir: Path, wanted: dict[str, tuple[str, int, int]] | None = None
) -> dict[str, Path]:
    """Copy embedded PNG outputs out of the committed notebooks.

    Args:
        notebooks_dir: Directory holding the ``.ipynb`` files.
        out_dir: Directory to write into. Created if absent.
        wanted: Figure name -> (notebook filename, cell index, output index). Defaults to
            :data:`NOTEBOOK_FIGURES`.

    Returns:
        Figure name -> the path written.

    Raises:
        FileNotFoundError: If a named notebook is missing.
        IndexError: If a cell or output index does not exist in that notebook.
        KeyError: If the addressed output carries no PNG.
    """
    wanted = NOTEBOOK_FIGURES if wanted is None else wanted
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    cache: dict[str, list] = {}
    for name, (notebook, cell_index, output_index) in wanted.items():
        if notebook not in cache:
            source = notebooks_dir / notebook
            if not source.exists():
                raise FileNotFoundError(f"notebook not found: {source}")
            cache[notebook] = json.loads(source.read_text(encoding="utf-8"))["cells"]
        cells = cache[notebook]

        if cell_index >= len(cells):
            raise IndexError(f"{notebook} has {len(cells)} cells; cell {cell_index} requested")
        outputs = cells[cell_index].get("outputs", [])
        if output_index >= len(outputs):
            raise IndexError(
                f"{notebook} cell {cell_index} has {len(outputs)} outputs; "
                f"output {output_index} requested — re-run the notebook or fix the address. "
                f"Cells carrying an image: {_png_addresses(cells)}"
            )
        data = outputs[output_index].get("data", {})
        if "image/png" not in data:
            raise KeyError(f"{notebook} cell {cell_index} output {output_index} carries no PNG")

        payload = data["image/png"]
        payload = payload if isinstance(payload, str) else "".join(payload)
        path = out_dir / name
        path.write_bytes(base64.b64decode(payload))
        written[name] = path

    return written


def extract_html_diagram(
    source: Path, out_dir: Path, name: str, width: int = DIAGRAM_WIDTH
) -> dict[str, Path]:
    """Lift the inline ``<svg>`` out of a hand-drawn HTML page and rasterize it.

    Same provenance rule as the notebook figures, for the same reason: the drawing is authored
    once, in the page, and both committed renderings are derived from it. A screenshot would be a
    third copy that nothing keeps honest.

    The page background is read from its own CSS rather than passed in, because the SVG paints no
    background of its own — several labels are knocked out against the page colour, and a PNG
    composited onto the wrong ground puts a visible patch behind them.

    Args:
        source: The ``.html`` page. Must carry exactly one ``<svg>`` element.
        out_dir: Directory to write into. Created if absent.
        name: Basename for the outputs, without extension.
        width: Raster width in pixels; height follows the viewBox aspect ratio.

    Returns:
        Extension (``"svg"``, ``"png"``) -> the path written. The PNG is absent if no rasterizer
        is installed — the SVG is the source of truth and is written either way.

    Raises:
        FileNotFoundError: If the page does not exist.
        ValueError: If it holds no ``<svg>``, more than one, or no usable ``viewBox``.
    """
    if not source.exists():
        raise FileNotFoundError(f"diagram page not found: {source}")
    page = source.read_text(encoding="utf-8")

    drawings = re.findall(r"<svg\b.*?</svg>", page, re.DOTALL)
    if len(drawings) != 1:
        raise ValueError(f"{source.name} holds {len(drawings)} <svg> elements; expected exactly 1")
    drawing = drawings[0]

    box = re.search(r'viewBox="\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*"', drawing)
    if box is None:
        raise ValueError(f"{source.name}'s <svg> has no numeric viewBox; cannot size the raster")
    inner_width, inner_height = float(box.group(3)), float(box.group(4))

    # Standalone files need an intrinsic size and the xlink namespace; inside HTML the browser
    # supplies both. Without them some rasterizers fall back to a 100x100 default.
    header = (
        f'<svg xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{inner_width:g}" height="{inner_height:g}" '
    )
    drawing = drawing.replace("<svg ", header, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    written["svg"] = out_dir / f"{name}.svg"
    written["svg"].write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + drawing, encoding="utf-8"
    )

    background = re.search(r"body\s*{[^}]*background:\s*(#[0-9a-fA-F]{3,8})", page)
    rasterizer = shutil.which("rsvg-convert")
    if rasterizer is None:
        # Reported, not raised: the SVG is on disk, and only the PNG embed is missing.
        print("  no rsvg-convert on PATH — PNG skipped. Install: brew install librsvg")
        return written

    written["png"] = out_dir / f"{name}.png"
    subprocess.run(
        [
            rasterizer,
            "--width",
            str(int(width)),
            "--background-color",
            background.group(1) if background else "white",
            "--output",
            str(written["png"]),
            str(written["svg"]),
        ],
        check=True,
    )
    return written


def cold_start_reach(
    development: pd.DataFrame, model_scores: pd.Series, k: int = FIRST_SCREEN
) -> dict[str, float]:
    """What share of deserving never-reviewed listings reaches a first screen.

    The report's sharpest negative result, and the one the headline metric cannot see. "Deserving"
    is the target's own judgement — grade 3 or above — so this asks whether the ranker surfaces
    the new listings its own training signal says belong near the top.

    The random figure is the analytic expectation ``min(k, n)/n`` averaged over the cohort's
    groups rather than a simulation, because the expectation is exact and a simulation would only
    add noise to a number the report quotes to one decimal.

    Args:
        development: The development pool, carrying ``query_group``, ``grade`` and
            ``number_of_reviews``.
        model_scores: Out-of-fold model score per row, aligned to ``development``.
        k: First-screen size.

    Returns:
        Ranker name -> the share of deserving cold-start listings it places in a top ``k``.

    Raises:
        ValueError: If no deserving cold-start listing exists to measure.
    """
    frame = development.copy()
    frame["model"] = model_scores
    frame["reviews"] = bl.rank_by_reviews(frame)

    worthy = frame["number_of_reviews"].eq(0) & frame["grade"].ge(3)
    if not worthy.any():
        raise ValueError("No never-reviewed listing graded 3 or above; nothing to measure.")

    grouped = frame.groupby("query_group", observed=True)
    sizes = grouped.size()
    reached = {
        "ideal (by grade)": grouped["grade"],
        "model": grouped["model"],
        "baseline (reviews)": grouped["reviews"],
    }
    shares = {
        name: float((worthy & column.rank(ascending=False, method="first").le(k)).sum())
        / float(worthy.sum())
        for name, column in reached.items()
    }
    shares["random (expected)"] = float(
        frame["query_group"].map(sizes.clip(upper=k) / sizes)[worthy].mean()
    )
    return shares


def _result_chart(sealed: pd.DataFrame, out_dir: Path) -> Path:
    """The headline, read against its floor rather than against zero."""
    overall = sealed.xs("overall", level="slice")
    seed = next(name for name in overall.index if str(name).startswith("model_seed"))

    scores = {
        "this model": float(overall.loc[seed, "ndcg@10"]),
        "rank by rating and price": float(overall.loc["price_rating", "ndcg@10"]),
        "rank by review count": float(overall.loc["reviews", "ndcg@10"]),
    }
    intervals = {
        "this model": (float(overall.loc[seed, "ndcg_low"]), float(overall.loc[seed, "ndcg_high"]))
    }
    figure, _ = eda.plot_scores_against_floor(
        scores,
        floor=float(overall.loc[seed, "floor"]),
        intervals=intervals,
        highlight="this model",
        title="NDCG@10 on the sealed fold, against the two frozen baselines",
        xlabel="NDCG@10 (1.0 = the best possible first screen)",
        floor_label="what a random shuffle scores",
    )
    path = out_dir / "result_vs_floor.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _cold_start_chart(shares: dict[str, float], out_dir: Path) -> Path:
    """The one chart where the model loses to chance."""
    random = shares["random (expected)"]
    figure, _ = eda.plot_scores_against_floor(
        {
            "model": shares["model"],
            "baseline (reviews)": shares["baseline (reviews)"],
        },
        floor=random,
        highlight="model",
        ceiling=shares["ideal (by grade)"],
        title="Deserving never-reviewed listings that reach a first screen",
        xlabel="share of the cohort reaching a top 10",
        floor_label="what a random shuffle reaches",
    )
    path = out_dir / "cold_start_reach.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _narrowing_chart(arms: pd.DataFrame, out_dir: Path) -> Path:
    """What narrowing the candidate set costs, and what it buys."""
    treatment = arms[arms["arm"].eq("treatment")].sort_values("k_geo")
    control = arms[arms["arm"].eq("control")].iloc[0]

    figure, _ = eda.plot_dose_response(
        [int(k) for k in treatment["k_geo"]],
        {
            "mean grade on the first screen": treatment["mean_grade"].tolist(),
            "deserving cold-start share": treatment["deserving_cold_share"].tolist(),
        },
        reference={
            "mean grade on the first screen": float(control["mean_grade"]),
            "deserving cold-start share": float(control["deserving_cold_share"]),
        },
        title="Narrowing the candidate set: what it costs, and what it buys",
        xlabel="neighbourhoods kept before ranking (k)",
        ylabels=["mean grade on the first screen", "deserving cold-start share"],
    )
    path = out_dir / "narrowing_tradeoff.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def main() -> None:
    """Write every figure the report embeds, and report what each chart is claiming."""
    out_dir = paths.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = extract_notebook_figures(paths.PROJECT_ROOT / "notebooks", out_dir)
    print(f"extracted {len(extracted)} notebook figures -> {out_dir}")
    for name in extracted:
        print(f"  {name}")

    for name, page in HTML_DIAGRAMS.items():
        rendered = extract_html_diagram(paths.PROJECT_ROOT / "docs" / page, out_dir, name)
        print(f"\nrendered {page} -> {', '.join(path.name for path in rendered.values())}")

    # One training run serves both model charts: the sealed table for the headline, and the
    # out-of-fold scores for the cohort analysis. Notebook 04 pays exactly this cost.
    print("\nrunning the training protocol for the two model charts ...")
    tables = trainer.run(log_to_mlflow=False)

    table = pd.read_parquet(paths.FEATURE_TABLE_PATH)
    fold, _ = split.assign_folds(table)
    development = table[~split.sealed_mask(fold)]

    result_path = _result_chart(tables["sealed"], out_dir)
    print(f"\nwritten: {result_path.name}")
    print(tables["sealed"].xs("overall", level="slice")[["ndcg@10", "floor"]].round(4).to_string())

    shares = cold_start_reach(development, tables["oof_scores"])
    cold_path = _cold_start_chart(shares, out_dir)
    print(f"\nwritten: {cold_path.name}")
    for name, share in sorted(shares.items(), key=lambda item: -item[1]):
        print(f"  {name:<22} {share:.4f}")

    arms_path = paths.TRAIN_DIR / "retrieval_arms.csv"
    if arms_path.exists():
        arms = pd.read_csv(arms_path)
        narrowing_path = _narrowing_chart(arms, out_dir)
        print(f"\nwritten: {narrowing_path.name}")
        print(
            arms[arms["arm"].eq("treatment")][["k_geo", "mean_grade", "deserving_cold_share"]]
            .round(4)
            .to_string(index=False)
        )
    else:
        # Reported, not raised: the other nine figures are already on disk and useful.
        print(
            f"\nSKIPPED narrowing_tradeoff.png — {arms_path} not found. "
            "Run: uv run python -m rental_ranking.evaluate.exposure"
        )


if __name__ == "__main__":
    main()

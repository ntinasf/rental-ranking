"""Tests for lifting a hand-drawn diagram out of its HTML page.

The one that carries the weight is ``test_the_committed_diagram_matches_its_page``: the whole
point of generating ``docs/figures/ab_traffic_split.*`` rather than screenshotting the page is
that the two cannot drift apart, and nothing enforces that unless a test re-extracts and compares.
Edit the page, forget to re-run the module, and this fails.

``test_every_notebook_address_still_resolves`` guards the other half of the module for the same
reason, one level cruder: those addresses are cell indices into notebooks nobody edits with this
table in mind, and a drifted one either crashes ``main()`` after a full training run or, worse,
resolves to the wrong picture.

The rest pin the extraction rules: a standalone SVG needs an intrinsic size the HTML original did
not (rasterizers fall back to a default without it), the page's own background colour has to reach
the PNG because several labels are knocked out against it, and an ambiguous page — no ``<svg>``, or
two — must raise rather than silently pick one.
"""

from pathlib import Path

import pytest

from rental_ranking import figures
from rental_ranking.data import paths

PAGE = """<!doctype html>
<title>a diagram</title>
<style>
  body { margin:0; background:#fbfaf8; color:#1c1a17; }
</style>
<div class="page">
<svg viewBox="0 0 1240 922" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" width="100" height="40"/>
  <text x="20" y="30">candidate set</text>
</svg>
<p class="note">Prose that must not travel with the drawing.</p>
</div>
"""


@pytest.fixture
def page(tmp_path: Path) -> Path:
    source = tmp_path / "diagram.html"
    source.write_text(PAGE, encoding="utf-8")
    return source


# --- the rule the module exists to keep ---------------------------------------------------------


def test_the_committed_diagram_matches_its_page(tmp_path: Path) -> None:
    """The committed SVG is what the page currently draws — re-extract and compare."""
    name, page = next(iter(figures.HTML_DIAGRAMS.items()))
    source = paths.PROJECT_ROOT / "docs" / page
    committed = paths.FIGURES_DIR / f"{name}.svg"
    if not source.exists():
        # The page is gitignored, so a clean clone has the rendering but not the drawing. Skip
        # rather than fail: there is nothing to compare against, which is not the same as a
        # mismatch. Un-ignoring the page is what would let this run everywhere.
        pytest.skip(f"{source} not present; the diagram's source page is not tracked")
    if not committed.exists():
        pytest.skip(f"{committed} not built; run: uv run python -m rental_ranking.figures")

    rebuilt = figures.extract_html_diagram(source, tmp_path, name)
    assert rebuilt["svg"].read_text(encoding="utf-8") == committed.read_text(encoding="utf-8"), (
        f"docs/{page} has changed since {committed.name} was written — "
        "re-run: uv run python -m rental_ranking.figures"
    )


def test_every_notebook_address_still_resolves(tmp_path: Path) -> None:
    """Each recorded (notebook, cell, output) still holds a PNG.

    Cell indices are brittle by nature — inserting one Markdown cell above a plot shifts every
    figure below it, and the notebook is edited far more often than this table. Left unchecked
    the drift surfaces as an IndexError from ``main()`` after a full training run, or worse, as a
    valid index pointing at the wrong picture.
    """
    notebooks = paths.PROJECT_ROOT / "notebooks"
    if not notebooks.exists():
        pytest.skip("notebooks/ not present")

    misses = []
    for figure, (notebook, cell, output) in figures.NOTEBOOK_FIGURES.items():
        try:
            figures.extract_notebook_figures(
                notebooks, tmp_path, {figure: (notebook, cell, output)}
            )
        except (IndexError, KeyError) as error:
            misses.append(f"{figure}: {error}")

    assert not misses, "\n".join(misses)


# --- extraction rules ---------------------------------------------------------------------------


def test_only_the_drawing_travels(page: Path, tmp_path: Path) -> None:
    """The page's prose and styling stay behind; the SVG is self-contained."""
    written = figures.extract_html_diagram(page, tmp_path / "out", "diagram")
    svg = written["svg"].read_text(encoding="utf-8")

    assert svg.startswith("<?xml")
    assert svg.rstrip().endswith("</svg>")
    assert "candidate set" in svg
    assert "must not travel" not in svg
    assert "<div" not in svg


def test_the_standalone_file_gains_an_intrinsic_size(page: Path, tmp_path: Path) -> None:
    """Inside HTML the browser sizes the SVG; alone it must say so itself, from the viewBox."""
    written = figures.extract_html_diagram(page, tmp_path / "out", "diagram")
    svg = written["svg"].read_text(encoding="utf-8")

    assert 'width="1240"' in svg
    assert 'height="922"' in svg
    assert 'viewBox="0 0 1240 922"' in svg


def test_a_page_without_a_drawing_raises(tmp_path: Path) -> None:
    """Silence here would write an empty figure the report would embed."""
    source = tmp_path / "empty.html"
    source.write_text("<!doctype html><p>no drawing at all</p>", encoding="utf-8")

    with pytest.raises(ValueError, match="0 <svg>"):
        figures.extract_html_diagram(source, tmp_path / "out", "empty")


def test_a_page_with_two_drawings_raises(tmp_path: Path) -> None:
    """Picking the first silently would make which figure you get depend on edit order."""
    source = tmp_path / "two.html"
    source.write_text(PAGE + PAGE, encoding="utf-8")

    with pytest.raises(ValueError, match="2 <svg>"):
        figures.extract_html_diagram(source, tmp_path / "out", "two")


def test_a_drawing_without_a_viewbox_raises(tmp_path: Path) -> None:
    """Without one there is no aspect ratio, so the raster height is unknowable."""
    source = tmp_path / "unsized.html"
    source.write_text('<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>', encoding="utf-8")

    with pytest.raises(ValueError, match="viewBox"):
        figures.extract_html_diagram(source, tmp_path / "out", "unsized")


def test_the_raster_carries_the_page_background(page: Path, tmp_path: Path) -> None:
    """The SVG paints no background of its own, and labels are knocked out against the page's."""
    pytest.importorskip("PIL")
    from PIL import Image

    written = figures.extract_html_diagram(page, tmp_path / "out", "diagram", width=124)
    if "png" not in written:
        pytest.skip("no rsvg-convert on PATH")

    with Image.open(written["png"]) as raster:
        # The viewBox aspect ratio, not a rasterizer default. Height is compared with a pixel of
        # slack because rsvg rounds a fractional height up.
        assert raster.width == 124
        assert abs(raster.height - 124 * 922 / 1240) <= 1
        assert raster.convert("RGB").getpixel((raster.width - 2, raster.height - 2)) == (
            0xFB,
            0xFA,
            0xF8,
        )

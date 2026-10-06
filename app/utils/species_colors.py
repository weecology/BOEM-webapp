"""Shared species → color system used across the map and every chart.

Colors are assigned once, by overall observation frequency across the full
(unfiltered) dataset, so a given species always gets the same color no matter
which page or filter is showing it — and a species keeps its color even when
a filter changes which species are currently visible.

Two different caps are used on purpose:
  - MAP_DISTINCT_LIMIT (3): on the map, many species can be visible on screen
    at once with every pairwise combination adjacent (a "scatter" context).
    Only the first 3 palette hues stay reliably distinguishable (including for
    colorblind viewers) when every pair can appear side by side, so species
    ranked 4+ fall back to the shared "Other species" gray on the map.
  - CHART_DISTINCT_LIMIT (8): bar charts only ever put *neighboring* bars
    next to each other, which the full 8-hue categorical palette clears. So
    charts can distinguish more species by color than the map can.

Species beyond a context's limit always render in OTHER_COLOR rather than a
generated/cycled hue — an unbounded species list (the taxonomy has 200+
entries) cannot each get a legible, distinct color, so the long tail is
grouped rather than assigned noise.
"""
import streamlit as st

# Fixed 8-hue categorical palette, validated for colorblind-safety in a fixed
# order (do not reorder or cycle — see the dataviz skill's color-formula doc).
CATEGORICAL_PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

OTHER_COLOR = "#898781"  # shared "Other species" gray — never a series hue
OTHER_LABEL = "Other species"

MAP_DISTINCT_LIMIT = 3
CHART_DISTINCT_LIMIT = 8


@st.cache_data
def build_species_rank(indices: dict) -> list:
    """Rank species (scientific name) by total observation count, most first.

    Computed once from the full prediction indices (not a filtered view) so
    a species' rank — and therefore its color — doesn't shift as a user
    applies filters on a page.
    """
    by_species = (indices or {}).get("by_species", {}) or {}
    return sorted(by_species, key=lambda sp: len(by_species[sp]), reverse=True)


def species_color(scientific: str, rank: list, limit: int = CHART_DISTINCT_LIMIT) -> str:
    """Fixed hue for a species within `limit`, else the shared 'Other' gray."""
    if not scientific or not rank:
        return OTHER_COLOR
    try:
        idx = rank.index(scientific)
    except ValueError:
        return OTHER_COLOR
    if idx >= limit or idx >= len(CATEGORICAL_PALETTE):
        return OTHER_COLOR
    return CATEGORICAL_PALETTE[idx]


def build_color_map(rank: list, limit: int = CHART_DISTINCT_LIMIT) -> dict:
    """species (scientific name) -> hex, for every ranked species."""
    return {sp: species_color(sp, rank, limit) for sp in rank}


def color_discrete_map(scientific_col, display_col, rank: list, limit: int = CHART_DISTINCT_LIMIT) -> dict:
    """Build a Plotly `color_discrete_map`: display label -> hex, covering
    exactly the labels present in a chart's data. Explicit per-label colors
    (rather than Plotly's auto-cycled palette) is what keeps a species the
    same color across every chart and the map.
    """
    colors = scientific_col.map(lambda s: species_color(s, rank, limit))
    return dict(zip(display_col, colors))


def render_color_legend(rank: list, limit: int, use_common: bool):
    """Render a compact swatch legend for the top-ranked species + 'Other'."""
    from utils.taxonomy import species_display

    if not rank:
        return
    top = rank[:limit]
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;'
        f'font-size:0.85rem;color:#374151;">'
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{species_color(sp, rank, limit)};margin-right:5px;"></span>'
        f'{species_display(sp, use_common)}</span>'
        for sp in top
    )
    swatches += (
        f'<span style="display:inline-flex;align-items:center;font-size:0.85rem;color:#374151;">'
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{OTHER_COLOR};margin-right:5px;"></span>{OTHER_LABEL}</span>'
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:4px 0;margin:0.35rem 0 0.75rem;">{swatches}</div>',
        unsafe_allow_html=True,
    )

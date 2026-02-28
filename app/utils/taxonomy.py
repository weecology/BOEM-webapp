"""Species name display: common (English) vs scientific names from taxonomy."""
from pathlib import Path
import json
from typing import Optional

import streamlit as st

# Taxonomy at project root (same level as app/)
_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "taxonomy.json"
# Fallback for app/data used by Bulk_Labeling
_TAXONOMY_PATH_ALT = Path(__file__).resolve().parents[1] / "data" / "taxonomy.json"

SPECIES_COLUMN_DISPLAY_NAME = "Species"


def _first_two_words(text: str) -> str:
    """Limit to first two words for shorter plot labels (e.g. 'Greater Shearwater, Great Shearwater' -> 'Greater Shearwater')."""
    if not text or not text.strip():
        return text
    words = text.split()
    return " ".join(words[:2]).rstrip(",").strip()


@st.cache_data
def _load_taxonomy_tree(path: Path) -> list:
    """Load taxonomy JSON. Returns list of root nodes."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _extract_species_mappings(nodes: list) -> dict[str, str]:
    """Recursively extract leaf species: scientificName -> common name (title before parenthesis)."""
    out = {}

    def visit(node: dict) -> None:
        if node.get("rank") == "Species":
            sci = node.get("scientificName") or ""
            title = node.get("title") or ""
            if sci and title and " (" in title:
                common = title.split(" (")[0].strip()
                if common:
                    out[sci] = common
        for child in node.get("children", []):
            visit(child)

    for root in nodes:
        visit(root)
    return out


@st.cache_data
def get_scientific_to_common(path: Optional[Path] = None) -> dict[str, str]:
    """Return mapping scientific name -> common name. Uses cached taxonomy."""
    p = path or _TAXONOMY_PATH
    if not p.exists():
        p = _TAXONOMY_PATH_ALT
    tree = _load_taxonomy_tree(p)
    return _extract_species_mappings(tree)


@st.cache_data
def get_display_to_scientific(path: Optional[Path] = None) -> dict[str, str]:
    """Map display name (first two words of common) back to scientific name for filtering.
    If multiple species share the same display name, the first encountered is used."""
    sci_to_common = get_scientific_to_common(path)
    out: dict[str, str] = {}
    for sci, common in sci_to_common.items():
        display = _first_two_words(common)
        if display and display not in out:
            out[display] = sci
    return out


def to_scientific(label: str) -> str:
    """Convert a label to scientific name for filtering. Pass-through if already scientific or unknown."""
    if not label or not str(label).strip():
        return label
    label = str(label).strip()
    sci_to_common = get_scientific_to_common()
    if label in sci_to_common:
        return label
    display_to_sci = get_display_to_scientific()
    return display_to_sci.get(label, label)


def species_display(scientific: str, use_common: bool = True) -> str:
    """Return display name for a species: common name or scientific name."""
    if scientific is None:
        return ""
    scientific = str(scientific).strip()
    if not scientific or scientific.lower() == "nan":
        return ""
    if use_common:
        mapping = get_scientific_to_common()
        common = mapping.get(scientific, scientific)
        return _first_two_words(common) if common else scientific
    return scientific

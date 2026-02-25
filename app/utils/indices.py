"""Load pre-computed prediction indices for faster Streamlit filtering."""
from pathlib import Path
import json
from typing import Optional

PREDICTIONS_INDICES_PATH = "app/data/predictions_indices.json"
EFFECTIVE_PREDICTIONS_PATH = "app/data/most_recent_all_flight_predictions_effective.csv"


def load_predictions_indices(path: str | Path = PREDICTIONS_INDICES_PATH) -> Optional[dict]:
    """Load predictions index (species_list, flight_list, by_species, by_flight). Returns None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

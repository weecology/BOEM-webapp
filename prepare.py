from app.utils import comet_utils
from app.utils.annotations import load_annotations, apply_annotations, ensure_human_labeled
from extract_coordinates import extract_flight_coordinates, generate_metadata
import pandas as pd

from pathlib import Path
import json
import os
import shutil

GULF_MAX_LONGITUDE = -80
IMAGES_DIR = Path("app/data/images")
MANIFEST_PATH = IMAGES_DIR / "crop_manifest.json"


def normalize_predictions_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize scores to 0-1 scale and add human_labeled column.

    - Adds human_labeled: True when set in train/validation/review, else False
    - For human_labeled rows (or score > 1): sets score = 1.0
    """
    df = df.copy()
    if 'set' not in df.columns:
        df['human_labeled'] = False
        return df
    df['human_labeled'] = df['set'].isin(['train', 'validation', 'review'])
    # Normalize scores: human-labeled and any score > 1 become 1.0
    mask = df['human_labeled'] | (df['score'] > 1)
    if 'score' in df.columns:
        df.loc[mask, 'score'] = 1.0
    return df


def build_predictions_indices(effective_df: pd.DataFrame) -> dict:
    """Build index dict: species_list, flight_list, by_species, by_flight."""
    effective_df = effective_df.copy()
    if 'crop_image_id' in effective_df.columns:
        effective_df['crop_image_id'] = effective_df['crop_image_id'].astype(str)
    species_list = sorted(effective_df['cropmodel_label'].dropna().unique().tolist())
    flight_list = (
        sorted(effective_df['flight_name'].dropna().unique().tolist())
        if 'flight_name' in effective_df.columns
        else []
    )
    by_species = (
        effective_df.groupby('cropmodel_label')['crop_image_id']
        .apply(lambda x: x.astype(str).tolist())
        .to_dict()
    )
    by_flight = {}
    if 'flight_name' in effective_df.columns:
        by_flight = (
            effective_df.groupby('flight_name')['crop_image_id']
            .apply(lambda x: x.astype(str).tolist())
            .to_dict()
        )
    return {
        'species_list': species_list,
        'flight_list': flight_list,
        'by_species': by_species,
        'by_flight': by_flight,
    }


def flight_basename(flight_name: str) -> str:
    """Strip the first underscore-delimited prefix (e.g. 'JPG') to get the
    metadata basename that matches .aflight / .csv filenames."""
    parts = flight_name.split("_")
    if len(parts) > 1:
        return "_".join(parts[1:])
    return flight_name


def get_gulf_flights(metadata_path: str = "app/data/metadata.csv") -> set:
    """Return metadata flight_name basenames whose mean longitude is in the
    Gulf of Mexico (west of GULF_MAX_LONGITUDE)."""
    if not os.path.exists(metadata_path):
        return set()
    metadata_df = pd.read_csv(metadata_path)
    mean_lon = metadata_df.groupby("flight_name")["long"].mean()
    return set(mean_lon[mean_lon < GULF_MAX_LONGITUDE].index)


def filter_gulf_predictions(predictions: pd.DataFrame,
                            gulf_basenames: set) -> pd.DataFrame:
    """Keep only predictions whose flight maps to a Gulf metadata basename."""
    basenames = predictions["flight_name"].apply(flight_basename)
    mask = basenames.isin(gulf_basenames)
    return predictions[mask].copy()


def load_manifest() -> dict:
    """Load the crop download manifest (previously downloaded experiments)."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(experiments: list):
    """Save the crop download manifest."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump({"experiments": sorted(experiments)}, f, indent=2)


def clear_crop_images():
    """Delete all crop images (but not the manifest) from the images directory."""
    if not IMAGES_DIR.exists():
        return
    for item in IMAGES_DIR.iterdir():
        if item.name == MANIFEST_PATH.name:
            continue
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


if __name__ == '__main__':
    # --- 1. Download newest metrics and predictions from Comet ---
    comet_utils.flight_model_metrics()
    comet_utils.detection_model_metrics()
    comet_utils.classification_model_metrics()

    # --- 2. Load raw latest predictions ---
    latest_predictions = pd.read_csv(
        "app/data/most_recent_all_flight_predictions.csv")

    # --- 3. Extract metadata for any new flights (needed for Gulf filter) ---
    for fn in latest_predictions['flight_name'].unique():
        fb = flight_basename(fn)
        if not os.path.exists(f"app/data/metadata/{fb}.csv"):
            extract_flight_coordinates(fb)
        else:
            print(f"Metadata already exists for {fn}")
    generate_metadata()

    # --- 4. Filter to Gulf of Mexico flights only ---
    gulf_basenames = get_gulf_flights()
    print(f"Gulf of Mexico flights (by metadata): {sorted(gulf_basenames)}")

    latest_predictions = filter_gulf_predictions(latest_predictions,
                                                 gulf_basenames)
    n_flights = latest_predictions['flight_name'].nunique()
    print(f"After Gulf filter: {len(latest_predictions)} predictions "
          f"across {n_flights} flight(s)")

    if latest_predictions.empty:
        print("WARNING: No Gulf flights found. "
              "Check metadata and .aflight files.")

    # Overwrite the CSV so the app only sees Gulf data
    latest_predictions.to_csv(
        "app/data/most_recent_all_flight_predictions.csv", index=False)

    # --- 5. Normalize scores and add human_labeled ---
    latest_predictions = normalize_predictions_scores(latest_predictions)
    latest_predictions.to_csv(
        "app/data/most_recent_all_flight_predictions.csv", index=False)

    # --- 6. Apply annotations and build indices ---
    annotations_df = load_annotations("app/data/annotations.csv")
    effective_predictions = apply_annotations(
        latest_predictions, annotations_df,
        id_col="crop_image_id", label_col="cropmodel_label", set_col="set"
    )
    effective_predictions = ensure_human_labeled(effective_predictions,
                                                 set_col="set")
    fp_mask = effective_predictions["cropmodel_label"] == "FalsePositive"
    fp_count = fp_mask.sum()
    if fp_count:
        print(f"Dropping {fp_count} FalsePositive rows from effective "
              "predictions")
    effective_predictions = effective_predictions[~fp_mask]
    effective_path = Path(
        "app/data/most_recent_all_flight_predictions_effective.csv")
    effective_predictions.to_csv(effective_path, index=False)
    indices = build_predictions_indices(effective_predictions)
    indices_path = Path("app/data/predictions_indices.json")
    with open(indices_path, "w") as f:
        json.dump(indices, f, indent=2)
    print(f"Wrote {indices_path} and {effective_path}")

    # --- 7. Normalize predictions.csv (full history) ---
    predictions_path = Path("app/data/predictions.csv")
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)
        predictions_df = normalize_predictions_scores(predictions_df)
        predictions_df.to_csv(predictions_path, index=False)

    # --- 8. Create shapefiles ---
    comet_utils.create_shapefiles(latest_predictions, "app/data/metadata.csv")

    # --- 9. Smart crop download ---
    current_experiments = sorted(
        latest_predictions['experiment'].unique().tolist())
    manifest = load_manifest()
    old_experiments = manifest.get("experiments", [])

    fp_image_ids = set(
        annotations_df.loc[
            annotations_df["new_label"] == "FalsePositive", "image_id"
        ].astype(str)
    )
    print(f"Will skip {len(fp_image_ids)} FalsePositive image(s) "
          "during download")

    if current_experiments == old_experiments:
        print("Crop manifest unchanged — skipping download")
        removed = 0
        for fp_id in fp_image_ids:
            fp_path = IMAGES_DIR / fp_id
            if fp_path.exists():
                fp_path.unlink()
                removed += 1
        if removed:
            print(f"Removed {removed} existing FalsePositive image(s) "
                  "from disk")
    else:
        print(f"Experiment set changed "
              f"({len(old_experiments)} → {len(current_experiments)}). "
              "Re-downloading crops...")
        clear_crop_images()

        for experiment_name in current_experiments:
            comet_utils.download_images(save_dir="app/data/images",
                                        experiment_name=experiment_name)
            exp_dir = IMAGES_DIR / experiment_name
            if exp_dir.exists():
                for image in os.listdir(str(exp_dir)):
                    if image in fp_image_ids:
                        os.remove(str(exp_dir / image))
                        continue
                    shutil.move(str(exp_dir / image),
                                str(IMAGES_DIR / image))
                shutil.rmtree(str(exp_dir))

        removed = 0
        for fp_id in fp_image_ids:
            fp_path = IMAGES_DIR / fp_id
            if fp_path.exists():
                fp_path.unlink()
                removed += 1
        if removed:
            print(f"Removed {removed} existing FalsePositive image(s) "
                  "from disk")

        save_manifest(current_experiments)
        print(f"Downloaded crops for {len(current_experiments)} experiment(s)")

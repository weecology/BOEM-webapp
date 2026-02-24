from app.utils import comet_utils
from app.utils.annotations import load_annotations, apply_annotations, ensure_human_labeled
from extract_coordinates import extract_flight_coordinates, generate_metadata
import pandas as pd

from pathlib import Path
import json
import os
import shutil


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


if __name__ == '__main__':
    # Download newest report
    comet_utils.flight_model_metrics()
    comet_utils.detection_model_metrics()
    comet_utils.classification_model_metrics()

    # Normalize scores and add human_labeled
    latest_predictions = pd.read_csv(
        "app/data/most_recent_all_flight_predictions.csv")
    latest_predictions = normalize_predictions_scores(latest_predictions)
    latest_predictions.to_csv("app/data/most_recent_all_flight_predictions.csv", index=False)

    # Apply annotations and build indices for faster Streamlit load
    annotations_df = load_annotations("app/data/annotations.csv")
    effective_predictions = apply_annotations(
        latest_predictions, annotations_df,
        id_col="crop_image_id", label_col="cropmodel_label", set_col="set"
    )
    effective_predictions = ensure_human_labeled(effective_predictions, set_col="set")
    effective_path = Path("app/data/most_recent_all_flight_predictions_effective.csv")
    effective_predictions.to_csv(effective_path, index=False)
    indices = build_predictions_indices(effective_predictions)
    indices_path = Path("app/data/predictions_indices.json")
    with open(indices_path, "w") as f:
        json.dump(indices, f, indent=2)
    print(f"Wrote {indices_path} and {effective_path}")

    # Normalize predictions.csv (full history)
    predictions_path = Path("app/data/predictions.csv")
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)
        predictions_df = normalize_predictions_scores(predictions_df)
        predictions_df.to_csv(predictions_path, index=False)

    # Lookup metadata for images
    for flight_name in latest_predictions['flight_name'].unique():
        flight_basename = "_".join(flight_name.split("_")[1:])
        # If metadata doesn't exist, process it
        if not os.path.exists(f"app/data/metadata/{flight_basename}.csv"):
            extract_flight_coordinates(flight_basename)
        else:
            print(f"Metadata already exists for {flight_name}")

    generate_metadata()
    comet_utils.create_shapefiles(latest_predictions, "app/data/metadata.csv")

    # Download images
    for experiment_name in latest_predictions['experiment'].unique():
        comet_utils.download_images(save_dir="app/data/images",
                                    experiment_name=experiment_name)
        # Combine all images into a single directory
        for image in os.listdir(f"app/data/images/{experiment_name}"):
            shutil.move(f"app/data/images/{experiment_name}/{image}",
                        f"app/data/images/{image}")
        # Remove the experiment directory
        shutil.rmtree(f"app/data/images/{experiment_name}")

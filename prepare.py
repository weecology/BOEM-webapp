from app.utils import comet_utils
from extract_coordinates import extract_flight_coordinates, generate_metadata
import pandas as pd

from pathlib import Path
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

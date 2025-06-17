from app.utils import comet_utils
from extract_coordinates import extract_flight_coordinates, generate_metadata
import pandas as pd

from pathlib import Path
import os

if __name__ == '__main__':
    # Download newest report    
    comet_utils.flight_model_metrics()
    comet_utils.detection_model_metrics()
    comet_utils.classification_model_metrics()
    
    # Create shapefiles
    latest_predictions = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")

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
    for experiment in latest_predictions['experiment'].unique():
        comet_utils.download_images(save_dir="app/static/images", experiment=experiment)

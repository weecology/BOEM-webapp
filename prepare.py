from app.utils import comet_utils
from extract_coordinates import extract_flight_coordinates, generate_metadata
import pandas as pd

from pathlib import Path
import os
import shutil

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
    for experiment_name in latest_predictions['experiment'].unique():
        comet_utils.download_images(save_dir="app/data/images", experiment_name=experiment_name)
        # Combine all images into a single directory
        for image in os.listdir(f"app/data/images/{experiment_name}"):
            shutil.move(f"app/data/images/{experiment_name}/{image}", f"app/data/images/{image}")
        # Remove the experiment directory
        shutil.rmtree(f"app/data/images/{experiment_name}")


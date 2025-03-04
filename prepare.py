from pathlib import Path
from app.utils.comet_utils import get_comet_experiments, download_images, create_shapefiles
import pandas as pd

if __name__ == '__main__':
    # Download newest report    
    get_comet_experiments()

    # Create shapefiles
    latest_predictions = pd.read_csv("app/data/processed/most_recent_all_flight_predictions.csv")
    create_shapefiles(latest_predictions, "app/data/processed/metadata.csv","app/data/processed/most_recent_all_flight_predictions.shp")

    # Download images
    latest_run = pd.read_csv("app/data/processed/most_recent_all_flight_predictions.csv")
    val_predictions = latest_run[latest_run['set'] == 'test']
    download_images(val_predictions)
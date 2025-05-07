from app.utils.comet_utils import get_comet_experiments, download_validation_images, create_shapefiles
import pandas as pd

if __name__ == '__main__':
    # Download newest report    
    get_comet_experiments()

    # Create shapefiles
    latest_predictions = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")
    create_shapefiles(latest_predictions, "app/data/metadata.csv")

    # Download images
    latest_run = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")
    for experiment in latest_run['experiment'].unique():
        download_validation_images(experiment)
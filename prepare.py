from app.utils import comet_utils
from extract_coordinates import extract_flight_coordinates, generate_metadata
import pandas as pd

# Download images from API paths
import requests
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

    # Create images directory if it doesn't exist
    image_dir = Path("app/data/images")
    image_dir.mkdir(parents=True, exist_ok=True)

    # Download images from API paths - limit to 100 per label
    for label in latest_predictions['cropmodel_label'].unique():
        # Get rows for this label and take first 100
        label_predictions = latest_predictions[latest_predictions['cropmodel_label'] == label].head(100)
        
        for _, row in label_predictions.iterrows():
            if pd.isna(row['crop_api_path']):
                continue
                
            image_path = image_dir / f"{row['crop_image_id']}"
            
            # Skip if image already exists
            if image_path.exists():
                continue
                
            try:
                response = requests.get(row['crop_api_path'])
                response.raise_for_status()
                with open(image_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Error downloading {row['crop_api_path']}: {e}")

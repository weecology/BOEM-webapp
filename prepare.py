from app.utils.comet_utils import get_comet_experiments, download_images, create_shapefiles
import pandas as pd

if __name__ == '__main__':
    # Download newest report    
    get_comet_experiments()

    # Create shapefiles
    latest_predictions = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")
    create_shapefiles(latest_predictions, "app/data/metadata.csv")

    # Download images'
    # Download images from API paths
    import requests
    from pathlib import Path
    import os

    # Create images directory if it doesn't exist
    image_dir = Path("app/data/images")
    image_dir.mkdir(parents=True, exist_ok=True)

    # Download images from API paths
    for _, row in latest_predictions.iterrows():
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


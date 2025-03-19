import os
from comet_ml import API
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from pathlib import Path
import geopandas as gpd
import json

# Load environment variables
load_dotenv()

def get_comet_experiments():
    """Get all experiments from the BOEM project with duration > 10min"""
    api = API(api_key=os.getenv('COMET_API_KEY'))
    workspace = os.getenv('COMET_WORKSPACE')
    
    # Get all experiments from the BOEM project
    experiments = api.get(f"{workspace}/boem")
    
    # Filter experiments by duration
    metrics_data = []
    all_predictions = []
    for exp in experiments:
        # Only 'pipeline' tags
        tags = exp.get_tags()
        if 'pipeline' not in tags:
            continue

        if 'complete' not in tags:
            continue

        if exp.archived:
            continue
        
        if exp.get_state() == 'running':
            continue
        
        duration = exp.get_metadata()['durationMillis']
        if duration > 600000:  # 10 minutes in milliseconds
            
            print(f"Processing experiment {exp.name}")

            # Get metrics
            metrics = exp.get_metrics()
            metrics_df = pd.DataFrame(metrics)
            metrics_df = metrics_df[metrics_df["metricName"].isin(["box_recall","box_precision"])]
            if metrics_df.empty:
                continue
            
            flight_name = exp.get_parameters_summary("flight_name")["valueCurrent"]
            
            # Latest value for each metric
            metrics_df = metrics_df.sort_values(by='timestamp', ascending=False).groupby('metricName').first().reset_index()
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'], unit='ms')
            metrics_df['experiment'] = exp.name
            metrics_df["flight_name"] = flight_name

            # Get final predictions
            final_predictions = exp.get_asset_by_name('final_predictions.csv')
            final_predictions = pd.read_csv(io.BytesIO(final_predictions))
            final_predictions["flight_name"] = flight_name
            final_predictions["experiment"] = exp.name
            final_predictions["timestamp"] = pd.to_datetime(exp.start_server_timestamp, unit='ms')
            
            # Combined and save predictions data
            all_predictions.append(final_predictions)
            metrics_data.append(metrics_df)

    all_predictions = pd.concat(all_predictions)
    all_predictions.to_csv("app/data/predictions.csv", index=False)

    # Write to csv
    pd.concat(metrics_data).to_csv("app/data/metrics.csv", index=False)

    # More recent prediction for each flight_name
    flight_dates = all_predictions.groupby('flight_name').agg({'timestamp': 'max'}).reset_index()
    most_recent_predictions = all_predictions[all_predictions.timestamp.isin(flight_dates["timestamp"])]
    most_recent_predictions.to_csv("app/data/most_recent_all_flight_predictions.csv", index=False)

def create_shapefiles(annotations, metadata):
    """Create shapefiles for each flight_name"""
    metadata_df = pd.read_csv(metadata)
    # Get the latest prediction for each flight_name 
    annotations["unique_image"] = annotations["image_path"].apply(lambda x: os.path.splitext(x)[0]).str.split("_").str.join("_")

    # All together as one shapefile
    merged_predictions = annotations.merge(metadata_df[["unique_image", "flight_name","date","lat","long"]], on='unique_image')
    gdf = gpd.GeoDataFrame(merged_predictions, geometry=gpd.points_from_xy(merged_predictions.long, merged_predictions.lat))
    gdf.to_file("app/data/all_predictions.shp", driver='ESRI Shapefile')

def download_validation_images(experiment, save_dir='app/data/images'):
    """Download all images logged to a Comet experiment"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    api = API(api_key=os.getenv('COMET_API_KEY'))
    workspace = os.getenv('COMET_WORKSPACE')
    
    # Get all experiments from the BOEM project
    experiment = api.get(f"{workspace}/boem",experiment=experiment)

    # Get all assets that are images
    assets = experiment.get_asset_list()
    image_assets = [a for a in assets if a['type'] == 'image']
  
    # Download each image
    image_data = []
    for asset in image_assets:
        if asset["metadata"] is None:
            continue

        # Get image name and data
        if not "validation" in asset["metadata"]:
            continue

        image_name = asset['fileName']
        image_path = save_dir / image_name
        
        # Only download if doesn't exist
        if not image_path.exists():
            image_data = experiment.get_asset(asset['assetId'])
            with open(image_path, 'wb') as f:
                f.write(image_data)
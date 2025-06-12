import os
from comet_ml import API
from dotenv import load_dotenv
import pandas as pd
import io
from pathlib import Path
import geopandas as gpd

# Load environment variables
load_dotenv()

def get_comet_metrics(metric_type='pipeline', output_file=None, metrics_to_track=None, include_predictions=False):
    """
    Get metrics from Comet.ml experiments
    
    Args:
        metric_type (str): Type of metrics to fetch ('pipeline', 'detection', or 'classification')
        output_file (str): Path to save the metrics CSV file
        metrics_to_track (list): List of metric names to track
        include_predictions (bool): Whether to include and process predictions
    
    Returns:
        tuple: (metrics_df, predictions_df) if include_predictions is True, else metrics_df
    """
    api = API(api_key=os.getenv('COMET_API_KEY'))
    workspace = os.getenv('COMET_WORKSPACE')
    
    # Get all experiments from the BOEM project
    experiments = api.get(f"{workspace}/boem")
    
    metrics_data = []
    all_predictions = []
    
    for exp in experiments:
        # Filter by tags
        tags = exp.get_tags()
        if metric_type not in tags:
            continue
            
        # Skip archived or running experiments
        if exp.archived or exp.get_state() == 'running':
            continue
            
        # Skip incomplete pipeline experiments
        if metric_type == 'pipeline' and 'complete' not in tags:
            continue
            
        # Skip preliminary flight
        if metric_type == 'pipeline':
            flight_name = exp.get_parameters_summary("flight_name")["valueCurrent"]
            if flight_name == "JPG_2024_Jan27":
                continue

        # Get metrics
        metrics = exp.get_metrics()
        metrics_df = pd.DataFrame(metrics)
        
        if metrics_to_track:
            metrics_df = metrics_df[metrics_df["metricName"].isin(metrics_to_track)]
            
        if not metrics_df.empty:
            # Get latest value for each metric
            metrics_df = metrics_df.sort_values(by='timestamp', ascending=False).groupby('metricName').first().reset_index()
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'], unit='ms')
            metrics_df['experiment'] = exp.name
            try:
                metrics_df["flight_name"] = exp.get_parameters_summary("flight_name")["valueCurrent"]
            except:
                pass
            metrics_data.append(metrics_df)
            
        # Process predictions if requested
        if include_predictions:
            try:
                final_predictions = exp.get_asset_by_name('final_predictions.csv', asset_type='dataframe')
                final_predictions = pd.read_csv(io.BytesIO(final_predictions))
                final_predictions["flight_name"] = exp.get_parameters_summary("flight_name")["valueCurrent"]
                final_predictions["experiment"] = exp.name
                final_predictions["timestamp"] = pd.to_datetime(exp.start_server_timestamp, unit='ms')
                all_predictions.append(final_predictions)
            except:
                pass

    # Combine and save metrics
    if metrics_data:
        metrics_df = pd.concat(metrics_data)
        if output_file:
            metrics_df.to_csv(output_file, index=False)
    
    # Process predictions if included
    if include_predictions and all_predictions:
        predictions_df = pd.concat(all_predictions)
        predictions_df.to_csv("app/data/predictions.csv", index=False)
        
        # Get latest predictions
        # Get the latest date for each flight_name
        latest_dates = predictions_df.groupby('flight_name')['timestamp'].max().reset_index()

        # Merge to get all predictions from the latest date for each flight_name
        latest_predictions = predictions_df.merge(latest_dates, on=['flight_name', 'timestamp'])
        
        latest_predictions.to_csv("app/data/most_recent_all_flight_predictions.csv", index=False)
        return metrics_df, predictions_df
    
    return metrics_df

def detection_model_metrics():
    """Get the metrics for the detection model"""
    return get_comet_metrics(
        metric_type='detection',
        output_file="app/data/detection_model_metrics.csv",
        metrics_to_track=["box_recall", "box_precision", "empty_frame_accuracy"]
    )

def classification_model_metrics():
    """Get the metrics for the classification model"""
    return get_comet_metrics(
        metric_type='classification',
        output_file="app/data/classification_model_metrics.csv",
        metrics_to_track=["Accuracy"]
    )

def flight_model_metrics():
    """Get all experiments from the BOEM project with duration > 10min"""
    return get_comet_metrics(
        metric_type='pipeline',
        output_file="app/data/metrics.csv",
        metrics_to_track=["box_recall", "box_precision", "empty_frame_accuracy"],
        include_predictions=True
    )

def create_shapefiles(annotations, metadata):
    """Create shapefiles for each flight_name"""
    metadata_df = pd.read_csv(metadata)
    # Get the latest prediction for each flight_name
    annotations["unique_image"] = annotations["image_path"].apply(lambda x: os.path.splitext(x)[0]).str.split("_").str.join("_")

    # All together as one shapefile
    merged_predictions = annotations.merge(metadata_df[["unique_image", "flight_name","date","lat","long"]], on='unique_image')
    gdf = gpd.GeoDataFrame(merged_predictions, geometry=gpd.points_from_xy(merged_predictions.long, merged_predictions.lat))
    gdf.crs = "EPSG:4326"
    gdf.to_file("app/data/all_predictions.shp", driver='ESRI Shapefile')

def download_images(experiment, save_dir='app/data/images'):
    """Download all images logged to a Comet experiment"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    api = API(api_key=os.getenv('COMET_API_KEY'))
    workspace = os.getenv('COMET_WORKSPACE')

    # Get all experiments from the BOEM project
    experiment = api.get(f"{workspace}/boem",experiment=experiment)

    # Get all assets that are images
    image_assets = experiment.get_asset_list(asset_type='image')

    # Download each image
    image_data = []
    for asset in image_assets:
        if asset["metadata"] is None:
            continue

        image_name = asset['fileName']
        image_path = save_dir / (image_name if image_name.endswith('.png') else f"{image_name}.png")

        # Only download if doesn't exist
        if not image_path.exists():
            image_data = experiment.get_asset(asset['assetId'])
            with open(image_path, 'wb') as f:
                f.write(image_data)

import os
from comet_ml import API
from dotenv import load_dotenv
import pandas as pd
import io
from pathlib import Path
import geopandas as gpd
import zipfile

# Load environment variables
load_dotenv()


def get_comet_metrics(metric_type='pipeline',
                      output_file=None,
                      metrics_to_track=None,
                      include_predictions=False):
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
            try: 
                flight_name = exp.get_parameters_summary(
                    "flight_name")["valueCurrent"]
            except:
                flight_name = None
            if flight_name == "JPG_2024_Jan27":
                continue

        # Get metrics
        metrics = exp.get_metrics()
        metrics_df = pd.DataFrame(metrics)

        if metrics_to_track:
            metrics_df = metrics_df[metrics_df["metricName"].isin(
                metrics_to_track)]

        if not metrics_df.empty:
            # Get latest value for each metric
            metrics_df = metrics_df.sort_values(
                by='timestamp',
                ascending=False).groupby('metricName').first().reset_index()
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'],
                                                     unit='ms')
            metrics_df['experiment'] = exp.name
            try:
                metrics_df["flight_name"] = exp.get_parameters_summary(
                    "flight_name")["valueCurrent"]
            except:
                pass
            metrics_data.append(metrics_df)

        # Process predictions if requested
        if include_predictions:
            try:
                final_predictions = exp.get_asset_by_name(
                    'final_predictions.csv', asset_type='dataframe')
                final_predictions = pd.read_csv(io.BytesIO(final_predictions))
                final_predictions["flight_name"] = exp.get_parameters_summary(
                    "flight_name")["valueCurrent"]
                final_predictions["experiment"] = exp.name
                final_predictions["timestamp"] = pd.to_datetime(
                    exp.start_server_timestamp, unit='ms')
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
        latest_dates = predictions_df.groupby(
            'flight_name')['timestamp'].max().reset_index()

        # Merge to get all predictions from the latest date for each flight_name
        latest_predictions = predictions_df.merge(
            latest_dates, on=['flight_name', 'timestamp'])

        latest_predictions.to_csv(
            "app/data/most_recent_all_flight_predictions.csv", index=False)
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
    """Create shapefiles for each flight_name.
    Expects annotations to include human_labeled column (from normalize_predictions_scores).
    Shapefile column names truncated to 10 chars: human_labeled -> human_lab
    """
    metadata_df = pd.read_csv(metadata)
    # Get the latest prediction for each flight_name
    annotations = annotations.copy()
    annotations["unique_image"] = annotations["image_path"].apply(lambda x: os.path.splitext(x)[0]).str.split("_").str.join("_")

    # All together as one shapefile
    metadata_df["unique_image"] = metadata_df["unique_image"].apply(lambda x: x.split("\\")[-1])
    merged_predictions = annotations.merge(metadata_df[["unique_image", "flight_name", "date", "lat", "long"]], on='unique_image')
    # Rename human_labeled to human_lab for shapefile 10-char column limit
    if "human_labeled" in merged_predictions.columns:
        merged_predictions = merged_predictions.rename(columns={"human_labeled": "human_lab"})
    gdf = gpd.GeoDataFrame(merged_predictions, geometry=gpd.points_from_xy(merged_predictions.long, merged_predictions.lat))
    gdf.crs = "EPSG:4326"
    gdf.to_file("app/data/all_predictions.shp", driver='ESRI Shapefile')

def download_images(experiment_name, save_dir='app/data/images'):
    """Download all images as crops.zip from a Comet experiment and unzip them"""
    save_dir = Path(save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    api = API(api_key=os.getenv('COMET_API_KEY'))
    workspace = os.getenv('COMET_WORKSPACE')

    # Get the experiment object
    experiment = api.get(f"{workspace}/boem", experiment=experiment_name)

    # Find crops.zip asset
    assets = experiment.get_asset_list()
    crops_zip_asset = next((a for a in assets if a['fileName'] == 'crops.zip'), None)
    if crops_zip_asset is None:
        print("No crops.zip found in experiment assets.")
        return

    # Download crops.zip
    zip_data = experiment.get_asset(crops_zip_asset['assetId'])
    zip_path = save_dir / f'{experiment.name}.zip'
    with open(zip_path, 'wb') as f:
        f.write(zip_data)

    # Unzip crops.zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    zip_path.unlink()  # Remove the zip file after extraction

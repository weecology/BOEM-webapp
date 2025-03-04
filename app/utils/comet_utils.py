import os
from comet_ml import API
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from pathlib import Path
import geopandas as gpd

# Load environment variables
load_dotenv()

def download_images(experiment, save_dir='app/data/images'):
    """Download all images logged to a Comet experiment"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all assets that are images
    assets = experiment.get_asset_list()
    image_assets = [a for a in assets if a['type'] == 'image']
    
    # Get test data for labels
    try:
        test_data = experiment.get_asset_by_name('test.csv')
        test_df = pd.read_csv(io.BytesIO(test_data))
    except:
        print(f"No test data found for experiment {experiment.name}")
        test_df = None
    
    # Download each image
    image_data = []
    for asset in image_assets:
        try:
            # Get image name and data
            image_name = asset['fileName']
            image_path = save_dir / image_name
            
            # Only download if doesn't exist
            if not image_path.exists():
                image_data = experiment.get_asset(asset['assetId'])
                with open(image_path, 'wb') as f:
                    f.write(image_data)
            
            # Get label if available
            if test_df is not None:
                label = test_df[test_df['image_path'].str.contains(image_name)]['label'].iloc[0]
            else:
                label = 'unknown'
                
            image_data.append({
                'image_path': str(image_path),
                'label': label,
                'experiment': experiment.name
            })
                
        except Exception as e:
            print(f"Error downloading image {image_name}: {str(e)}")
            
    return pd.DataFrame(image_data) if image_data else None

def get_comet_experiments():
    """Get all experiments from the BOEM project with duration > 10min"""
    api = API(api_key=os.getenv('COMET_API_KEY'))
    workspace = os.getenv('COMET_WORKSPACE')
    
    # Get all experiments from the BOEM project
    experiments = api.get(f"{workspace}/boem")
    
    # Filter experiments by duration
    metrics_data = []
    label_counts_data = []
    
    all_predictions = []
    for exp in experiments:
        duration = exp.get_metadata()['durationMillis']
        if duration > 600000:  # 10 minutes in milliseconds
            
            # Get metrics
            metrics = exp.get_metrics()
            metrics_df = pd.DataFrame(metrics)
            metrics_df = metrics_df[metrics_df["metricName"].isin(["detection_box_recall","detection_box_precision"])]
            if metrics_df.empty:
                continue
            
            # Latest value for each metric
            metrics_df = metrics_df.sort_values(by='timestamp', ascending=False).groupby('metricName').first().reset_index()
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'], unit='ms')
            metrics_df['experiment'] = exp.name
            
            # Get train and test data
            train_data = exp.get_asset_by_name('train.csv')
            train_df = pd.read_csv(io.BytesIO(train_data))
            train_df['experiment'] = exp.name
            train_df['timestamp'] = pd.to_datetime(exp.start_server_timestamp, unit='ms')
            train_df["set"] = "train"
            
            train_counts = train_df['label'].value_counts().reset_index()
            train_counts.columns = ['type', 'count']
            train_counts['set'] = 'train'
            train_counts['experiment'] = exp.name
            train_counts['timestamp'] = pd.to_datetime(exp.start_server_timestamp, unit='ms')
            label_counts_data.append(train_counts)
            
            # Get test data
            test_data = exp.get_asset_by_name('test.csv')
            test_df = pd.read_csv(io.BytesIO(test_data))
            test_df['experiment'] = exp.name
            test_df['timestamp'] = pd.to_datetime(exp.start_server_timestamp, unit='ms')
            test_df["set"] = "test"
            
            test_counts = test_df['label'].value_counts().reset_index()
            test_counts.columns = ['type', 'count']
            test_counts['set'] = 'test'
            test_counts['experiment'] = exp.name
            test_counts['timestamp'] = pd.to_datetime(exp.start_server_timestamp, unit='ms')
            label_counts_data.append(test_counts)
            test_counts["set"] = "test"

            # Get active learning data
            active_learning_data = exp.get_asset_by_name('training_pool_predictions.csv')
            if active_learning_data is not None:
                active_learning_df = pd.read_csv(io.BytesIO(active_learning_data))
                active_learning_df['experiment'] = exp.name
                active_learning_df['timestamp'] = pd.to_datetime(exp.start_server_timestamp, unit='ms')
                active_learning_df["set"] = "predictions"
            
            # Get human reviewed data
            human_reviewed_data = exp.get_asset_by_name('human_reviewed_annotations.csv')
            if human_reviewed_data is not None:
                human_reviewed_df = pd.read_csv(io.BytesIO(human_reviewed_data))
                human_reviewed_df['experiment'] = exp.name
                human_reviewed_df['timestamp'] = pd.to_datetime(exp.start_server_timestamp, unit='ms')
                human_reviewed_df["set"] = "human_reviewed"

            # Combined and save predictions data
            experiment_predictions = pd.concat([train_df, test_df, active_learning_df, human_reviewed_df])
            all_predictions.append(experiment_predictions)
            metrics_data.append(metrics_df)

    all_predictions = pd.concat(all_predictions)
    all_predictions.to_csv("app/data/processed/predictions.csv", index=False)

    # Write to csv
    pd.concat(metrics_data).to_csv("app/data/processed/metrics.csv", index=False)
    pd.concat(label_counts_data).to_csv("app/data/processed/label_counts.csv", index=False)

    # More recent prediction for each flight_name
    most_recent_predictions = all_predictions.groupby('flight_name').apply(lambda x: x.loc[x['timestamp'].idxmax()])
    most_recent_predictions.to_csv("app/data/processed/most_recent_all_flight_predictions.csv", index=False)

def create_label_count_plots(label_counts_df):
    """Create plots showing label distributions over time"""
    if label_counts_df is None:
        return None, None
        
    # Create time series plot of label counts
    fig_timeseries = px.line(
        label_counts_df,
        x='timestamp',
        y='count',
        color='type',
        line_dash='set',
        title='Label Counts Over Time by Set',
        labels={'count': 'Number of Instances', 'type': 'Label Type'}
    )
    
    # Create histogram of most recent model's counts
    latest_exp = label_counts_df['timestamp'].max()
    latest_counts = label_counts_df[label_counts_df['timestamp'] == latest_exp]
    
    fig_hist = px.bar(
        latest_counts,
        x='type',
        y='count',
        color='set',
        title=f'Label Distribution in Latest Model ({pd.to_datetime(latest_exp).strftime("%Y-%m-%d")})',
        labels={'type': 'Label Type', 'count': 'Number of Instances', 'set': 'Dataset'},
        barmode='group'
    )
    
    return fig_timeseries, fig_hist

def create_metric_plots(metrics_df):
    """Create plots showing metric improvement over time"""
    # Line plot for each metric over time
    fig_metrics = px.line(
        metrics_df,
        x='timestamp',
        y='value',
        color='experiment',
        facet_col='metric',
        title='Metrics Over Time'
    )
    
    return fig_metrics

def create_prediction_plots(predictions_df):
    """Create plots showing prediction distributions"""
    if predictions_df is None:
        return None
        
    # Count predictions by type and experiment
    pred_counts = predictions_df.groupby(['timestamp', 'type']).size().reset_index(name='count')
    
    fig_preds = px.bar(
        pred_counts,
        x='timestamp',
        y='count',
        color='type',
        title='Predictions by Type',
        barmode='group'
    )
    
    return fig_preds 

def create_shapefiles(annotations, metadata):
    """Create shapefiles for each flight_name"""
    metadata_df = pd.read_csv(metadata)
    # Get the latest prediction for each flight_name 
    latest_predictions = annotations.groupby('flight_name').apply(lambda x: x.loc[x['timestamp'].idxmax()])
    for flight_name in latest_predictions['flight_name'].unique():
        # Connect with metadata on location
        merged_predictions = latest_predictions.merge(metadata_df[["unique_image", "flight_name","date","lat","long"]], on='unique_image')

        # Create shapefile
        gdf = gpd.GeoDataFrame(merged_predictions, geometry=gpd.points_from_xy(merged_predictions.long, merged_predictions.lat))
        gdf.to_file(f"app/data/processed/shapefiles/{flight_name}.shp", driver='ESRI Shapefile')
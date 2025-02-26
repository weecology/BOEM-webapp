import os
from comet_ml import API
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Load environment variables
load_dotenv()

def get_comet_experiments():
    """Get all experiments from the BOEM project with duration > 10min"""
    api = API(api_key=os.getenv('COMET_API_KEY'))
    workspace = os.getenv('COMET_WORKSPACE')
    
    # Get all experiments from the BOEM project
    experiments = api.get(f"{workspace}/boem")
    
    # Filter experiments by duration
    long_experiments = []
    metrics_data = []
    predictions_data = []
    
    for exp in experiments:
        duration = exp.get_metadata()['durationMillis']
        if duration > 600000:  # 10 minutes in milliseconds
            long_experiments.append(exp)
            
            # Get metrics
            metrics = exp.get_metrics()
            metrics_df = pd.DataFrame(metrics)
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'], unit='ms')
            metrics_df['experiment'] = exp.name
            metrics_data.append(metrics_df)
            
            # Get predictions if they exist
            try:
                predictions = exp.get_asset('predictions.csv')
                pred_df = pd.read_csv(predictions)
                pred_df['experiment'] = exp.name
                pred_df['timestamp'] = exp.get_metadata()['created']
                predictions_data.append(pred_df)
            except:
                print(f"No predictions found for experiment {exp.name}")
    
    return pd.concat(metrics_data), pd.concat(predictions_data) if predictions_data else None

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
    pred_counts = predictions_df.groupby(['experiment', 'type']).size().reset_index(name='count')
    
    fig_preds = px.bar(
        pred_counts,
        x='experiment',
        y='count',
        color='type',
        title='Predictions by Type and Experiment',
        barmode='group'
    )
    
    return fig_preds 
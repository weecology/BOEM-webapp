import streamlit as st
from utils.styling import load_css
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Model Development",
    page_icon="📈",
    layout="wide"
)

def create_metric_plots(metrics_df):
    """Create plots showing metric improvement over time"""
    # Line plot for each metric over time
    fig_metrics = px.line(
        metrics_df,
        x='timestamp',
        y='metricValue',
        facet_col='metricName',
        title='Metrics Over Time'
    )
    
    return fig_metrics

def create_prediction_plots(predictions_df):
    """Create plots showing prediction distributions"""
    if predictions_df is None:
        return None
    
    # Convert timestamp to datetime
    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
    
    # Calculate total counts per timestamp
    total_counts = predictions_df.groupby('timestamp').size()
    
    # Calculate percentage for each label at each timestamp
    label_percentages = predictions_df.groupby(['timestamp', 'cropmodel_label']).size().unstack(fill_value=0)
    label_percentages = label_percentages.div(total_counts, axis=0) * 100
    
    # Melt the dataframe for plotting
    plot_df = label_percentages.reset_index().melt(
        id_vars=['timestamp'],
        var_name='Species',
        value_name='Percentage'
    )
    
    # Create line plot
    fig = px.line(
        plot_df,
        x='timestamp',
        y='Percentage',
        color='Species',
        title='Species Composition Over Time',
        labels={
            'timestamp': 'Date',
            'Percentage': 'Percentage of Total Predictions',
            'Species': 'Species'
        }
    )
    
    fig.update_layout(
        height=600,
        xaxis_tickangle=-45,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig, label_percentages

def app():
    st.title("Model Development Metrics")
    metrics_df = pd.read_csv("app/data/metrics.csv")

    #Flight name dropdown
    flight_name = st.selectbox("Select Flight Name", metrics_df["flight_name"].unique())

    # Get experiment data
    metrics_df = pd.read_csv("app/data/metrics.csv")

    # Filter metrics and predictions for selected flight name
    metrics_df = metrics_df[metrics_df["flight_name"] == flight_name]

    # Create and display metrics plot
    st.subheader("Training Metrics")
    metrics_plot = create_metric_plots(metrics_df)
    st.plotly_chart(metrics_plot, use_container_width=True)

    # Display raw data in expandable sections
    with st.expander("View Raw Metrics Data"):
        st.dataframe(metrics_df[['metricName', 'metricValue', 'timestamp', 'flight_name']])
    
    # Create and display predictions plot
    st.subheader("Species Composition")
    predictions_df = pd.read_csv("app/data/predictions.csv")
    predictions_plot, percentages_df = create_prediction_plots(predictions_df)
    st.plotly_chart(predictions_plot, use_container_width=True)
    
    # Display the percentage data
    with st.expander("View Species Percentages Data"):
        st.dataframe(
            percentages_df.style.format("{:.2f}%"),
            height=400
        )

if __name__ == "__main__":
    load_css()
    app()
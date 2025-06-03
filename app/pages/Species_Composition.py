import streamlit as st
from utils.styling import load_css
import pandas as pd
import plotly.express as px

def create_prediction_plots(predictions_df):
    """Create plots showing prediction distributions"""
    if predictions_df is None:
        return None, None, None
    
    # Convert timestamp to datetime
    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
    
    # Calculate total counts per timestamp
    total_counts = predictions_df.groupby('timestamp').size()
    
    # Calculate percentage for each label at each timestamp
    label_percentages = predictions_df.groupby(['timestamp', 'cropmodel_label']).size().unstack(fill_value=0)
    label_percentages = label_percentages.div(total_counts, axis=0) * 100
    
    # Melt the dataframes for plotting
    plot_df_percent = label_percentages.reset_index().melt(
        id_vars=['timestamp'],
        var_name='Species',
        value_name='Percentage'
    )
    
    # Create raw counts dataframe
    raw_counts = predictions_df.groupby(['timestamp', 'cropmodel_label']).size().reset_index(name='count')
    
    # Create percentage plot
    fig_percent = px.line(
        plot_df_percent,
        x='timestamp',
        y='Percentage',
        color='Species',
        title='Species Composition Over Time (Percentage)',
        labels={
            'timestamp': 'Date',
            'Percentage': 'Percentage of Total Predictions',
            'Species': 'Species'
        }
    )
    
    fig_percent.update_layout(
        height=600,
        xaxis_tickangle=-45,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    
    # Create raw counts plot
    fig_raw = px.line(
        raw_counts,
        x='timestamp',
        y='count',
        color='cropmodel_label',
        title='Species Composition Over Time (Raw Counts)',
        labels={
            'timestamp': 'Date',
            'count': 'Number of Detections',
            'cropmodel_label': 'Species'
        }
    )
    
    fig_raw.update_layout(
        height=600,
        xaxis_tickangle=-45,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig_percent, fig_raw, label_percentages

def app():
    st.title("Species Composition Analysis")
    
    # Read the data
    predictions_df = pd.read_csv("app/data/predictions.csv")
    
    # Detection score slider
    detection_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01
    )
    
    # Add flight name dropdown
    flight_name = st.selectbox("Select Flight Name", predictions_df["flight_name"].unique())
    
    # Filter data for selected flight
    filtered_df = predictions_df[predictions_df["flight_name"] == flight_name]
    
    # Filter by detection score if column exists
    if 'score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['score'] >= detection_threshold]
    
    # Create and display predictions plot
    predictions_plot_percent, predictions_plot_raw, percentages_df = create_prediction_plots(filtered_df)
    
    st.plotly_chart(predictions_plot_percent, use_container_width=True)
    st.plotly_chart(predictions_plot_raw, use_container_width=True)
    
    # Display the percentage data
    with st.expander("View Species Percentages Data"):
        st.dataframe(
            percentages_df.style.format("{:.2f}%"),
            height=400
        )

if __name__ == "__main__":
    load_css()
    app() 
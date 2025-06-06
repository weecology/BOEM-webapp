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
    
    # Create raw counts dataframe
    raw_counts = predictions_df.groupby(['timestamp', 'cropmodel_label']).size().reset_index(name='count')
    
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
    
    return fig_raw

def app():
    st.title("Species Composition Analysis")
    st.text("The species composition analysis shows how the predictions of species change over time during model development and annotation. This addresses the question of whether additional iterations of model development will lead to changes in downstream data.")
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
    
    # Create and display raw count plot
    raw_plot = create_prediction_plots(filtered_df)
    st.plotly_chart(raw_plot, use_container_width=True)
    
    # Add rare species plot below
    species_counts = filtered_df['cropmodel_label'].value_counts()
    if not species_counts.empty:
        max_count = species_counts.iloc[0]
        rare_threshold = max_count * 0.10
        rare_species = species_counts[species_counts < rare_threshold]
        if not rare_species.empty:
            rare_df = filtered_df[filtered_df['cropmodel_label'].isin(rare_species.index)]
            rare_counts = rare_df['cropmodel_label'].value_counts().reset_index()
            rare_counts.columns = ['label', 'count']
            rare_fig = px.bar(
                rare_counts,
                x='label',
                y='count',
                title='Rare Species (<10% of Most Common)',
                labels={'label': 'Species', 'count': 'Number of Instances'}
            )
            st.plotly_chart(rare_fig, use_container_width=True)
        else:
            st.info('No rare species (less than 10% of the most common) found in this dataset.')
    else:
        st.info('No species data available.')

if __name__ == "__main__":
    load_css()
    app() 
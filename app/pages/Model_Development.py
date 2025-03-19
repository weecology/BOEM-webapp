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

def create_comet_links(experiment_id):
    """Create HTML links to different Comet dashboard views"""
    base_url = "https://www.comet.com/bw4sz/boem/"
    
    links = {
        'Detection Images': f"{base_url}{experiment_id}?experiment-tab=images&groupBy=metadata%25_context&orderBy=desc&sortBy=metadata%25_context",
        'Confusion Matrix': f"{base_url}{experiment_id}?experiment-tab=confusionMatrix",
        'Loss Curves': f"{base_url}{experiment_id}?experiment-tab=charts",
        'Classification Images': f"{base_url}{experiment_id}?experiment-tab=images&groupBy=metadata%25_context&orderBy=desc&sortBy=metadata%25_context&query=classification"
    }
    
    # Create HTML links
    html_links = {
        k: f'<a href="{v}" target="_blank">{k}</a>'
        for k, v in links.items()
    }
    
    return html_links

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

    # Create experiment links table at the bottom
    st.subheader("Comet.ml Dashboard Links")
    
    # Get unique experiments
    experiments = metrics_df[['experiment', 'flight_name']].drop_duplicates()
    
    # Create links for each experiment
    links_data = []
    for _, row in experiments.iterrows():
        links = create_comet_links(row['experiment'])
        links_data.append({
            'Flight': row['flight_name'],
            'Experiment ID': row['experiment'],
            **links
        })
    
    # Create DataFrame with links
    links_df = pd.DataFrame(links_data)
    
    # Display as HTML table
    st.write(
        links_df.to_html(
            escape=False,
            index=False
        ),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    load_css()
    app()
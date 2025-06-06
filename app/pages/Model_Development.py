import streamlit as st
from utils.styling import load_css
import pandas as pd
import plotly.express as px

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
    st.text("Machine models in collaboration with the University of Florida. There are two models: a classification model and a detection model. The detection model identifies objects of interest, and the classification model classifies them the finest level of detail possible. The model for each flight starts from a backbone model trained on the entire dataset and additional data from the ecological monitoring community")
    st.title("Backbone Models")

    st.header("Detection Model")
    # --- Detection Model Metrics Section ---
    try:
        detection_metrics_df = pd.read_csv("app/data/detection_metrics.csv")
        st.subheader("Detection Model Metrics Over Time")
        # Plot detection metrics
        if not detection_metrics_df.empty:
            fig_detection = px.line(
                detection_metrics_df,
                x='timestamp',
                y='metricValue',
                color='metricName',
                title='Detection Model Metrics Over Time',
                labels={'timestamp': 'Timestamp', 'metricValue': 'Metric Value', 'metricName': 'Metric'}
            )
            st.plotly_chart(fig_detection, use_container_width=True)
            # Show table
            with st.expander("View Detection Model Metrics Data"):
                st.dataframe(detection_metrics_df)
        else:
            st.info("No detection model metrics available.")
    except Exception as e:
        st.warning(f"Could not load detection model metrics: {e}")

    st.header("Classification Model")

    st.title("Flight Model Metrics")
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
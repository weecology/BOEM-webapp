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
    
    fig_metrics.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
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
    st.header("Model Development")
    st.text("Machine learning models are developed in collaboration with the University of Florida. The detection model identifies objects of interest, and the classification model classifies them the finest level of detail possible. The model for each flight starts from a backbone model trained on the entire dataset and additional data from the ecological monitoring community. It is then customized for each flight.")

    st.header("Detection Backbone")
    # --- Detection Model Metrics Section ---
    try:
        detection_metrics_df = pd.read_csv("app/data/detection_model_metrics.csv")
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
            fig_detection.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02
                )
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
    # --- Classification Model Metrics Section ---
    try:
        classification_metrics_df = pd.read_csv("app/data/classification_model_metrics.csv")
        st.subheader("Classification Model Metrics Over Time")
        # Plot classification metrics
        if not classification_metrics_df.empty:
            fig_classification = px.line(
                classification_metrics_df,
                x='timestamp',
                y='metricValue',
                color='metricName',
                title='Classification Model Metrics Over Time',
                labels={'timestamp': 'Timestamp', 'metricValue': 'Metric Value', 'metricName': 'Metric'}
            )
            fig_classification.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02
                )
            )
            st.plotly_chart(fig_classification, use_container_width=True)
            # Show table
            with st.expander("View Classification Model Metrics Data"):
                st.dataframe(classification_metrics_df)
        else:
            st.info("No classification model metrics available.")
    except Exception as e:
        st.warning(f"Could not load classification model metrics: {e}")

    # --- Latest Flight Metrics Comparison ---
    st.header("Latest Flight Metrics")
    try:
        metrics_df = pd.read_csv("app/data/metrics.csv")
        if not metrics_df.empty:
            # Get the latest metrics for each flight and metric
            latest_metrics = metrics_df.sort_values('timestamp').groupby(['flight_name', 'metricName']).last().reset_index()
            
            # Create bar chart
            fig_latest = px.bar(
                latest_metrics,
                x='flight_name',
                y='metricValue',
                color='metricName',
                title='Latest Metrics by Flight',
                labels={
                    'flight_name': 'Flight',
                    'metricValue': 'Metric Value',
                    'metricName': 'Metric'
                },
                barmode='group'
            )
            fig_latest.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_tickangle=-45,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02
                )
            )
            st.plotly_chart(fig_latest, use_container_width=True)
            
            # Show table
            with st.expander("View Latest Metrics Data"):
                st.dataframe(latest_metrics)
        else:
            st.info("No flight metrics available.")
    except Exception as e:
        st.warning(f"Could not load flight metrics: {e}")

    st.header("Flight Model Metrics")
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

    # --- Species Composition Analysis (moved from Species_Composition.py) ---
    st.header("Species Composition Analysis")
    st.text("The species composition analysis shows how the predictions of species change over time during model development and annotation. This addresses the question of whether additional iterations of model development will lead to changes in downstream data.")
    predictions_df = pd.read_csv("app/data/predictions.csv")
    
    detection_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01
    )
    
    flight_name = st.selectbox("Select Flight Name (Species Comp)", predictions_df["flight_name"].unique())
    filtered_df = predictions_df[predictions_df["flight_name"] == flight_name]
    if 'score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['score'] >= detection_threshold]
    
    # Create and display raw count plot
    def create_prediction_plots(predictions_df):
        if predictions_df is None:
            return None
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        raw_counts = predictions_df.groupby(['timestamp', 'cropmodel_label']).size().reset_index(name='count')
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
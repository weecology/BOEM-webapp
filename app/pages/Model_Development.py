import streamlit as st
from utils.styling import load_css
import pandas as pd
import plotly.express as px
from utils.annotations import load_annotations, apply_annotations, ensure_human_labeled
from utils.indices import load_predictions_indices
from utils.taxonomy import species_display


@st.cache_data
def _load_indices():
    return load_predictions_indices()

def app():
    st.header("Model Development")
    st.text("Machine learning models are developed in collaboration with the University of Florida. The detection model identifies objects of interest, and the classification model classifies them to the finest level of detail possible. A single backbone model is trained on the entire dataset and additional data from the ecological monitoring community.")
    show_trend = st.checkbox("Show trend line on metric plots", value=False, help="Add a linear (OLS) trend line to scatter plots; minimal performance impact.")

    st.header("Detection Backbone")
    # --- Detection Model Metrics Section ---
    try:
        detection_metrics_df = pd.read_csv("app/data/detection_model_metrics.csv")
        # Exclude 0.0 values from aborted runs so y-axis scales to real data
        detection_metrics_df = detection_metrics_df[detection_metrics_df["metricValue"] != 0]
        st.subheader("Detection Model Metrics Over Time")
        # Plot detection metrics
        if not detection_metrics_df.empty:
            detection_metrics_df = detection_metrics_df.copy()
            detection_metrics_df["timestamp"] = pd.to_datetime(detection_metrics_df["timestamp"])
            fig_detection = px.scatter(
                detection_metrics_df,
                x='timestamp',
                y='metricValue',
                color='metricName',
                title='Detection Model Metrics Over Time',
                labels={'timestamp': 'Timestamp', 'metricValue': 'Metric Value', 'metricName': 'Metric'},
                trendline='ols' if show_trend else None,
            )
            fig_detection.update_traces(marker=dict(size=10))
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
        # Exclude 0.0 values from aborted runs so y-axis scales to real data
        classification_metrics_df = classification_metrics_df[classification_metrics_df["metricValue"] != 0]
        st.subheader("Classification Model Metrics Over Time")
        # Only Micro-averaged accuracy in the main figure
        micro_accuracy_df = classification_metrics_df[
            classification_metrics_df["metricName"] == "Micro-Average Accuracy"
        ]
        if not micro_accuracy_df.empty:
            micro_accuracy_df = micro_accuracy_df.copy()
            micro_accuracy_df["timestamp"] = pd.to_datetime(micro_accuracy_df["timestamp"])
            fig_classification = px.scatter(
                micro_accuracy_df,
                x='timestamp',
                y='metricValue',
                title='Classification Model Metrics Over Time',
                labels={'timestamp': 'Timestamp', 'metricValue': 'Metric Value'},
                trendline='ols' if show_trend else None,
            )
            fig_classification.update_traces(marker=dict(size=10))
            fig_classification.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_classification, use_container_width=True)
        else:
            st.info("No Micro-Average Accuracy data available.")
        # Species / per-class metric: dropdown to pick any metric and plot over time
        st.subheader("Species or per-class classification metric")
        all_metric_names = sorted(classification_metrics_df["metricName"].unique().tolist())
        if all_metric_names:
            selected_metric = st.selectbox(
                "Select metric to plot over time",
                options=all_metric_names,
                key="classification_metric_select",
                help="e.g. Class Accuracy_Alle alle for per-species accuracy over time",
            )
            metric_series = classification_metrics_df[
                classification_metrics_df["metricName"] == selected_metric
            ].sort_values("timestamp")
            if not metric_series.empty:
                metric_series = metric_series.copy()
                metric_series["timestamp"] = pd.to_datetime(metric_series["timestamp"])
                fig_species = px.scatter(
                    metric_series,
                    x="timestamp",
                    y="metricValue",
                    title=f"{selected_metric} over time",
                    labels={"timestamp": "Timestamp", "metricValue": "Metric Value"},
                    trendline='ols' if show_trend else None,
                )
                fig_species.update_traces(marker=dict(size=10))
                fig_species.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_species, use_container_width=True)
            else:
                st.info(f"No data for {selected_metric}.")
        # Show table for all classification metrics
        with st.expander("View Classification Model Metrics Data"):
            st.dataframe(classification_metrics_df)
    except Exception as e:
        st.warning(f"Could not load classification model metrics: {e}")

    # --- Species Composition Analysis (moved from Species_Composition.py) ---
    use_common = st.session_state.get("use_common_names", True)
    st.header("Species Composition Analysis")
    st.text("The species composition analysis shows how the predictions of species change over time during model development and annotation. This addresses the question of whether additional iterations of model development will lead to changes in downstream data.")
    predictions_df = pd.read_csv("app/data/predictions.csv")
    annotations_df = load_annotations("app/data/annotations.csv")
    predictions_df = apply_annotations(predictions_df, annotations_df, id_col="crop_image_id", label_col="cropmodel_label", set_col="set")
    predictions_df = ensure_human_labeled(predictions_df, set_col="set")

    detection_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01,
        help="Filter by model detection confidence (0-1 scale)"
    )
    human_labeled_only = st.checkbox(
        "Human-labeled only (Species Comp)",
        value=False,
        help="Show only predictions that have been reviewed by a human"
    )
    indices = _load_indices()
    flight_options = indices["flight_list"] if indices else sorted(predictions_df["flight_name"].unique().tolist())
    flight_name = st.selectbox("Select Flight Name (Species Comp)", flight_options)
    filtered_df = predictions_df[predictions_df["flight_name"] == flight_name]
    if 'score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['score'] >= detection_threshold]
    if human_labeled_only:
        filtered_df = filtered_df[filtered_df['human_labeled'] == True]
    
    # Create and display raw count plot (use display names for Species)
    def create_prediction_plots(predictions_df, use_common_names=True):
        if predictions_df is None:
            return None
        predictions_df = predictions_df.copy()
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        predictions_df['Species'] = predictions_df['cropmodel_label'].map(lambda s: species_display(s, use_common_names))
        raw_counts = predictions_df.groupby(['timestamp', 'Species']).size().reset_index(name='count')
        fig_raw = px.scatter(
            raw_counts,
            x='timestamp',
            y='count',
            color='Species',
            title='Species Composition Over Time (Raw Counts)',
            labels={
                'timestamp': 'Date',
                'count': 'Number of Detections',
                'Species': 'Species'
            },
        )
        fig_raw.update_traces(marker=dict(size=10))
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
    raw_plot = create_prediction_plots(filtered_df, use_common)
    st.plotly_chart(raw_plot, use_container_width=True)
    # Add rare species plot below (display names for Species)
    species_counts = filtered_df['cropmodel_label'].value_counts()
    if not species_counts.empty:
        max_count = species_counts.iloc[0]
        rare_threshold = max_count * 0.10
        rare_species = species_counts[species_counts < rare_threshold]
        if not rare_species.empty:
            rare_df = filtered_df[filtered_df['cropmodel_label'].isin(rare_species.index)]
            rare_counts = rare_df['cropmodel_label'].value_counts().reset_index()
            rare_counts.columns = ['label', 'count']
            rare_counts['Species'] = rare_counts['label'].map(lambda s: species_display(s, use_common))
            rare_fig = px.bar(
                rare_counts,
                x='Species',
                y='count',
                title='Rare Species (<10% of Most Common)',
                labels={'Species': 'Species', 'count': 'Number of Instances'}
            )
            st.plotly_chart(rare_fig, use_container_width=True)
        else:
            st.info('No rare species (less than 10% of the most common) found in this dataset.')
    else:
        st.info('No species data available.')

if __name__ == "__main__":
    load_css()
    app()
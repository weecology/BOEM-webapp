import streamlit as st
from utils.styling import load_css
import pandas as pd
import plotly.express as px
from utils.taxonomy import species_display, get_scientific_to_common


def app():
    st.header("Model Development")
    st.text("Machine learning models are developed in collaboration with the University of Florida. The detection model identifies objects of interest, and the classification model classifies them to the finest level of detail possible. A single backbone model is trained on the entire dataset and additional data from the ecological monitoring community.")
    show_trend = st.checkbox("Show trend line on metric plots", value=False, help="Add a linear (OLS) trend line to scatter plots; minimal performance impact.")

    st.header("Detection Backbone")
    # --- Detection Model Metrics Section ---
    ZERO_SHOT_METRICS = {
        "zero_shot_evaluation_box_precision",
        "zero_shot_evaluation_box_recall",
        "zero_shot_evaluation_empty_frame_accuracy",
    }
    try:
        detection_metrics_df = pd.read_csv("app/data/detection_model_metrics.csv")
        # Exclude 0.0 values from aborted runs so y-axis scales to real data
        detection_metrics_df = detection_metrics_df[detection_metrics_df["metricValue"] != 0]
        st.subheader("Detection Model Metrics Over Time")
        # In-flight metrics only (exclude zero-shot so they appear in the separate section)
        in_flight_df = detection_metrics_df[
            ~detection_metrics_df["metricName"].isin(ZERO_SHOT_METRICS)
        ].copy()
        if not in_flight_df.empty:
            in_flight_df["timestamp"] = pd.to_datetime(in_flight_df["timestamp"])
            fig_detection = px.scatter(
                in_flight_df,
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
            st.caption("These metrics are evaluated on data from the same flight(s) used in training or validation.")
            # Show table
            with st.expander("View Detection Model Metrics Data"):
                st.dataframe(detection_metrics_df)
        else:
            st.info("No detection model metrics available.")

        # --- Zero-Shot Evaluation Metrics (held-out flights) ---
        st.subheader("Zero-Shot Evaluation Metrics Over Time")
        st.markdown(
            "The metrics above use data from the same flight. The zero-shot metrics below are computed on **held-out flights** to test the ability of the model to generalize through time."
        )
        zero_shot_df = detection_metrics_df[
            detection_metrics_df["metricName"].isin(ZERO_SHOT_METRICS)
        ].copy()
        if not zero_shot_df.empty:
            zero_shot_df["timestamp"] = pd.to_datetime(zero_shot_df["timestamp"])
            fig_zero_shot = px.scatter(
                zero_shot_df,
                x='timestamp',
                y='metricValue',
                color='metricName',
                title='Zero-Shot Evaluation Metrics Over Time',
                labels={'timestamp': 'Timestamp', 'metricValue': 'Metric Value', 'metricName': 'Metric'},
                trendline='ols' if show_trend else None,
            )
            fig_zero_shot.update_traces(marker=dict(size=10))
            fig_zero_shot.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02
                )
            )
            st.plotly_chart(fig_zero_shot, use_container_width=True)
        else:
            st.info("No zero-shot evaluation metrics available. Newer detection runs may log zero_shot_evaluation_box_precision, zero_shot_evaluation_box_recall, and zero_shot_evaluation_empty_frame_accuracy.")
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
            use_size = "val_support" in micro_accuracy_df.columns
            fig_classification = px.scatter(
                micro_accuracy_df,
                x='timestamp',
                y='metricValue',
                size='val_support' if use_size else None,
                size_max=20,
                title='Classification Model Metrics Over Time',
                labels={'timestamp': 'Timestamp', 'metricValue': 'Accuracy'},
                trendline='ols' if show_trend else None,
                hover_data=['val_support'] if use_size else None,
            )
            if not use_size:
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
        sci_to_common = get_scientific_to_common()

        def metric_display_name(metric_name: str) -> str:
            """Convert e.g. 'Class Accuracy_Larus argentatus' to 'Herring Gull (Larus argentatus)'."""
            prefix = "Class Accuracy_"
            if metric_name.startswith(prefix):
                scientific = metric_name[len(prefix):].strip()
                common = sci_to_common.get(scientific, scientific)
                return f"{common} ({scientific})"
            return metric_name

        if all_metric_names:
            display_to_metric = {metric_display_name(m): m for m in all_metric_names}
            display_options = [metric_display_name(m) for m in all_metric_names]
            selected_display = st.selectbox(
                "Select metric to plot over time",
                options=display_options,
                key="classification_metric_select",
                help="Per-species accuracy over time. Shown as English name (Scientific name).",
            )
            selected_metric = display_to_metric[selected_display]
            metric_series = classification_metrics_df[
                classification_metrics_df["metricName"] == selected_metric
            ].sort_values("timestamp")
            if not metric_series.empty:
                metric_series = metric_series.copy()
                metric_series["timestamp"] = pd.to_datetime(metric_series["timestamp"])
                use_size = "val_support" in metric_series.columns
                fig_species = px.scatter(
                    metric_series,
                    x="timestamp",
                    y="metricValue",
                    size="val_support" if use_size else None,
                    size_max=20,
                    title=f"{selected_display} over time",
                    labels={"timestamp": "Timestamp", "metricValue": "Accuracy"},
                    trendline='ols' if show_trend else None,
                    hover_data=["val_support"] if use_size else None,
                )
                if not use_size:
                    fig_species.update_traces(marker=dict(size=10))
                fig_species.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_species, use_container_width=True)
            else:
                st.info(f"No data for {selected_display}.")
        # Show table for all classification metrics
        with st.expander("View Classification Model Metrics Data"):
            st.dataframe(classification_metrics_df)
    except Exception as e:
        st.warning(f"Could not load classification model metrics: {e}")

if __name__ == "__main__":
    load_css()
    app()
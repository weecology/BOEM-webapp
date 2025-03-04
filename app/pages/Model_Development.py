import streamlit as st
from utils.styling import load_css
import pandas as pd
from utils.comet_utils import get_comet_experiments, create_metric_plots, create_label_count_plots

st.set_page_config(
    page_title="Model Development",
    page_icon="📈",
    layout="wide"
)

def app():
    st.title("Model Development Metrics")
    
    #Flight name dropdown
    flight_name = st.selectbox("Select Flight Name", metrics_df["flight_name"].unique(),default=metrics_df["flight_name"].unique()[0])

    # Get experiment data
    metrics_df = pd.read_csv("app/data/processed/metrics.csv")
    predictions_df = pd.read_csv(f"app/data/processed/predictions.csv")
    label_counts_df = pd.read_csv("app/data/processed/label_counts.csv")

    # Filter metrics and predictions for selected flight name
    metrics_df = metrics_df[metrics_df["flight_name"] == flight_name]
    predictions_df = predictions_df[predictions_df["flight_name"] == flight_name]

    # Create and display metrics plot
    st.subheader("Training Metrics")
    metrics_plot = create_metric_plots(metrics_df)
    st.plotly_chart(metrics_plot, use_container_width=True)
    
    # Create and display label distribution plots
    if label_counts_df is not None:
        st.subheader("Label Distribution Analysis")
        time_plot, hist_plot = create_label_count_plots(label_counts_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(time_plot, use_container_width=True)
        with col2:
            st.plotly_chart(hist_plot, use_container_width=True)
        
        # Display raw data in expandable sections
        with st.expander("View Raw Metrics Data"):
            st.dataframe(metrics_df)
        
        with st.expander("View Raw Label Counts Data"):
            st.dataframe(label_counts_df)


if __name__ == "__main__":
    load_css()
    app()
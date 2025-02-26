import streamlit as st
from utils.styling import load_css
from utils.comet_utils import get_comet_experiments, create_metric_plots, create_prediction_plots

st.set_page_config(
    page_title="Model Development",
    page_icon="📈",
    layout="wide"
)

def app():
    st.title("Model Development Metrics")
    
    try:
        # Get experiment data
        with st.spinner('Loading experiment data from Comet.ml...'):
            metrics_df, predictions_df = get_comet_experiments()
        
        # Create and display metrics plot
        st.subheader("Training Metrics")
        metrics_plot = create_metric_plots(metrics_df)
        st.plotly_chart(metrics_plot, use_container_width=True)
        
        # Create and display predictions plot
        if predictions_df is not None:
            st.subheader("Prediction Distribution")
            pred_plot = create_prediction_plots(predictions_df)
            st.plotly_chart(pred_plot, use_container_width=True)
            
            # Display raw data in expandable sections
            with st.expander("View Raw Metrics Data"):
                st.dataframe(metrics_df)
            
            with st.expander("View Raw Predictions Data"):
                st.dataframe(predictions_df)
    
    except Exception as e:
        st.error(f"Error loading experiment data: {str(e)}")
        st.info("Please ensure your Comet.ml API key and workspace are properly configured in the .env file")

if __name__ == "__main__":
    load_css()
    app()
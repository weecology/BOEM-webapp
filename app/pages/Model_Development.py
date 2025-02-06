import streamlit as st
import pandas as pd

report = pd.read_csv("app/data/report.csv")
detection_url = report.loc[0, "Detection URL"]
classification_url = report.loc[0, "Classification URL"]

st.title("Model Development")

st.header("Detection")

# Filter for detection metrics
detection_metrics = report.filter(regex='^detection_').reset_index(drop=True)

# Clean up column names by removing 'detection_' prefix
detection_metrics.columns = detection_metrics.columns.str.replace('detection_', '')

# Create two column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Detection Metrics:")
    # Format metrics as table
    st.dataframe(
        detection_metrics.T.rename(columns={0: 'Value'})
        .style.format("{:.3f}")
    )

st.markdown(f"For more details, see the [Detection Report Confusion Matrix]({detection_url + '?experiment-tab=confusionMatrix'})")

st.header("Species Classification") 

st.markdown(f"For more details, see the [Classification Report Confusion Matrix]({classification_url + '?experiment-tab=confusionMatrix'})")

st.subheader("Species Classification Accuracy:")

st.subheader("Species Classification Confusion Matrix:")
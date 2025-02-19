import streamlit as st
import pandas as pd

report = pd.read_csv("app/data/report.csv")
#detection_url = report.loc[0, "Detection URL"]
detection_url = "https://www.comet.com/bw4sz/boem/fe70626847ec4b948b9d40db82a5e6be?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step"
#classification_url = report.loc[0, "Classification URL"]
classification_url = "https://www.comet.com/bw4sz/boem/96c2e80648c9497181738264230aae6b?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step"
st.title("Model Development")

st.header("Model Overview")

# Create two column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Details")
    st.markdown(f"""
    - **Model**: {report.loc[0, 'model_name']}
    - **Total Images**: {int(report.loc[0, 'total_images']):,}
    - **Total Annotations**: {int(report.loc[0, 'total_annotations'])}
    - **Human Reviewed**: {int(report.loc[0, 'human_reviewed_images'])}
    - **Completion Rate**: {report.loc[0, 'completion_rate']:.2%}
    """)

with col2:
    st.subheader("Prediction Stats")
    detection_map = float(report.loc[0, 'detection_map'].replace('tensor(', '').replace(')', ''))
    st.markdown(f"""
    - **Confident Predictions**: {int(report.loc[0, 'confident_predictions'])}
    - **Uncertain Predictions**: {int(report.loc[0, 'uncertain_predictions'])}
    - **Detection mAP**: {detection_map:.3f}
    """)

st.header("Classification Performance")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Confident Predictions")
    confident_accuracy = float(report.loc[0, 'confident_classification_accuracy'].replace('tensor(', '').replace(')', ''))
    st.metric(
        label="Classification Accuracy",
        value=f"{confident_accuracy:.1%}"
    )

with col4:
    st.subheader("Uncertain Predictions") 
    uncertain_accuracy = float(report.loc[0, 'uncertain_classification_accuracy'].replace('tensor(', '').replace(')', ''))
    st.metric(
        label="Classification Accuracy",
        value=f"{uncertain_accuracy:.1%}"
    )

st.markdown(f"For more details, see the [Classification Report Confusion Matrix]({classification_url + '?experiment-tab=confusionMatrix'})")
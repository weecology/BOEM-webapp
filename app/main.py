import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path
import leafmap.foliumap as leafmap
import pandas as pd
import geopandas as gpd
import os
from PIL import Image
from utils.auth import require_login
from utils.annotations import load_annotations, apply_annotations, apply_annotations_to_gdf, ensure_human_labeled

st.set_page_config(
    page_title="Bureau of Ocean Energy Management - Gulf of America Biodiversity Survey",
    page_icon="🦅",
    layout="wide"
)

# Require login for all content
require_login()

# Read the data
data_path = Path(__file__).parent / "data" / "most_recent_all_flight_predictions.csv"

if not data_path.exists():
    st.error(f"File not found: {data_path}")
    st.stop()

# Add the app directory to Python path
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

df = pd.read_csv(data_path)

# Apply overrides to predictions table
annotations_df = load_annotations("app/data/annotations.csv")
df = apply_annotations(df, annotations_df, id_col="crop_image_id", label_col="cropmodel_label", set_col="set")

df = df.loc[df.score>0.7]

# Only keep two word labels
df = df[df["cropmodel_label"].str.count(" ") == 1]
gdf = gpd.read_file(data_path.parent / "all_predictions.shp")
# Apply overrides to geodataframe (join on crop_image)
gdf = apply_annotations_to_gdf(gdf, annotations_df, gdf_image_col="crop_image", gdf_label_col="cropmodel_", gdf_set_col="set")
gdf['date'] = pd.to_datetime(gdf['date'], errors='coerce')

st.title("Bureau of Ocean Energy Management Biodiversity Survey")
st.text("Use this tool to visualize biodiversity data collected during aerial surveys of offshore energy development areas. The tool uses AI to detect and classify marine wildlife species in aerial images. These data are used to inform the development of offshore projects using rapid and cost-effective airborne surveys.")

# Show the conceptual figure from app/data/conceptual_figure.png
conceptual_figure = Image.open("app/www/conceptual_figure.png")
st.image(conceptual_figure, caption="Raw image from a flight with detections and classifications overlaid for bottlenose dolphins", width=700, use_container_width='always')

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        ### Overview
        
        This project processes aerial survey data to:
        - Detect and classify marine wildlife species
        - Generate distribution maps and abundance estimates
        - Analyze temporal and spatial patterns
        - Support environmental impact assessments

        The data viewer includes:
        - Interactive maps for viewing survey tracks and observations
        - Analysis tools for exploring species distributions
        - Image galleries of detected species
    """)

with col2:
    # Display basic statistics
    # Statistics have a min score of 0.7
    st.subheader("Progress")
    st.write(f"Total Observations: {len(df)}")
    df_labeled = ensure_human_labeled(df, "set")
    st.write(
        f"Human-reviewed observations: {df_labeled['human_labeled'].sum()}")
    st.write(f"Species: {df['cropmodel_label'].nunique()}")
    st.write(f"Aerial Surveys: {df['flight_name'].nunique()}")

# Add detection backbone model metrics
try:
    st.subheader("Performance") 
    st.write(
        "To evaluate the performance of this workflow, we track the performance using images not use to train the models. A model's recall is the proportion of true biodiversity objects detected by the model. A model precision is the proportion of predictions that are true biodiversity objects."
    )
    detection_metrics_df = pd.read_csv(
        "app/data/detection_model_metrics.csv")
    if not detection_metrics_df.empty:
        latest_metrics = detection_metrics_df.sort_values(
            'timestamp').groupby('metricName').last()
        st.subheader("Detection Model")
        if 'box_recall' in latest_metrics.index:
            st.write(
                f"Recall: {latest_metrics.loc['box_recall', 'metricValue'] * 100:.1f}%"
            )
        if 'box_precision' in latest_metrics.index:
            st.write(
                f"Precision: {latest_metrics.loc['box_precision', 'metricValue'] * 100:.1f}%"
            )
        if 'empty-frame-precision' in latest_metrics.index:
            st.write(
                f"Empty Frame Precision: {latest_metrics.loc['empty-frame-precision', 'metricValue'] * 100:.1f}%"
            )
        if 'empty_frame_accuracy' in latest_metrics.index:
            st.write(
                f"Empty Frame Accuracy: {latest_metrics.loc['empty_frame_accuracy', 'metricValue'] * 100:.1f}%"
            )
except:
    pass

# Add classification backbone model metrics
try:
    classification_metrics_df = pd.read_csv(
        "app/data/classification_model_metrics.csv")
    if not classification_metrics_df.empty:
        # Get the most recent metrics for each requested metric
        latest_metrics = classification_metrics_df.sort_values(
            'timestamp').groupby('metricName').last()
        st.subheader("Classification Model")
        if 'Accuracy' in latest_metrics.index:
            st.write(
                f"Accuracy: {latest_metrics.loc['Accuracy', 'metricValue']:.3f}"
            )
        if 'Precision' in latest_metrics.index:
            st.write(
                f"Precision: {latest_metrics.loc['Precision', 'metricValue']:.3f}"
            )
        # Add HTML link to the latest classification experiment's Confusion Matrix on Comet
        try:
            latest_experiment = classification_metrics_df.sort_values(
                'timestamp').iloc[-1]['experiment']
            comet_confusion_url = f"https://www.comet.com/bw4sz/boem/{latest_experiment}?experiment-tab=confusionMatrix"
            st.markdown(
                f'<a href="{comet_confusion_url}" target="_blank">View Classification Confusion Matrix</a>',
                unsafe_allow_html=True)
        except Exception:
            pass
except:
    pass

st.header("Observations")

# Add download button for all predictions
try:
    predictions_df = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="Download All Predictions Data",
        data=csv,
        file_name="all_predictions.csv",
        mime="text/csv",
        help="Download the most recent predictions for all flights"
    )
except:
    pass

m = leafmap.Map(center=[40, -70], zoom=6)
app_data_dir = Path(__file__).parent / "data"
default_file = app_data_dir / "all_predictions.shp"
if default_file.exists():
    gdf_obs = gpd.read_file(default_file)
    gdf_obs = gdf_obs[gdf_obs['cropmodel_'].notna()]
    annotations_df = load_annotations("app/data/annotations.csv")
    gdf_obs = apply_annotations_to_gdf(gdf_obs, annotations_df, gdf_image_col="crop_image", gdf_label_col="cropmodel_", gdf_set_col="set")
    gdf_obs = ensure_human_labeled(gdf_obs, set_col="set")
    predictions_df = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")
    unique_labels = sorted(gdf_obs['cropmodel_'].unique())
    score_threshold = st.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Filter observations by detection confidence (0-1 scale)"
    )
    selected_labels = st.multiselect(
        "Species",
        options=unique_labels,
        default=["Tursiops truncatus","Delphinidae"],
        help="Select species to display"
    )
    human_reviewed = st.checkbox(
        "Only Human-labeled images",
        value=True,
        help="If checked, only images that have been reviewed by a human will be shown."
    )
    m.add_basemap("OpenStreetMap")
    m.add_wms_layer(
        url="https://wms.gebco.net/mapserv?",
        layers="GEBCO_LATEST",
        name="GEBCO Bathymetry",
        format="image/png",
        transparent=True,
        attribution="GEBCO"
    )
    try:
        filtered_gdf = gdf_obs[
            (gdf_obs['score'] >= score_threshold) &
            (gdf_obs['cropmodel_'].isin(selected_labels))
        ]
        if human_reviewed:
            filtered_gdf = filtered_gdf[filtered_gdf['human_labeled'] == True]
        if len(filtered_gdf) == 0:
            st.warning("No observations meet the selected filters")
        else:
            # Remove image_link/crop_api_path logic
            filtered_gdf = filtered_gdf[['score', 'cropmodel_', 'cropmode_1', 'set',
                            'flight_n_1', 'date', 'lat', 'long', 'crop_image', 'geometry']].rename(columns={
                'score': 'Detection Score',
                'cropmodel_': 'Species',
                'cropmode_1': 'Species Confidence',
                'set': 'Set',
                'flight_n_1': 'Flight Name',
                'date': 'Date',
                'lat': 'Latitude',
                'long': 'Longitude',
                'crop_image': 'Image'
            })

            m.add_gdf(
                filtered_gdf,
                layer_name=f"Observations (score ≥ {score_threshold})",
                style={'color': "#0000FF"},
                info_mode='on_click',
                hover_style={'sticky': True},
                info_columns=['Image']
            )
    except Exception as e:
        st.error(f"Error processing vector data: {str(e)}")
    m.to_streamlit(height=700, width=None)
    # Gallery of up to 20 images below the map
    gallery_images = filtered_gdf['Image'].dropna().unique()[:20]
    if len(gallery_images) > 0:
        st.subheader("Selected Observations")
        cols = st.columns(3)
        for idx, img in enumerate(gallery_images):
            img_path = Path("app/data/images") / str(img)
            with cols[idx % 3]:
                if img_path.exists():
                    # Add confidence score to caption
                    confidence_score = filtered_gdf[filtered_gdf['Image'] == img]['Species Confidence'].values[0]
                    st.image(str(img_path), caption=f"{img} (Confidence: {confidence_score:.2f})", use_container_width=True)
                else:
                    st.info(f"Image {img} not found.")
    else:
        st.info("No images available for the selected observations.")

    # Download button for filtered data
    try:
        temp_file = app_data_dir / "temp_filtered.shp"
        filtered_gdf.to_file(temp_file)
        with open(temp_file, 'rb') as file:
            shapefile_bytes = file.read()
            st.download_button(
                label="Download Data",
                data=shapefile_bytes,
                file_name=f"predictions_filtered.shp",
                mime="application/octet-stream",
                help=f"Download filtered observations (score ≥ {score_threshold})"
            )
        temp_file.unlink()
    except Exception as e:
        st.error(f"Error preparing download: {str(e)}")
else:
    st.error(f"Data file not found: {default_file}")

st.info("""
**How to use:**
- Click on any point on the map to view its metadata
- Use the confidence score slider to filter observations
- Select species to show/hide from the map
""")

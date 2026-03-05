import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path
import folium
import leafmap.foliumap as leafmap
import pandas as pd
import geopandas as gpd
import os
from PIL import Image
from utils.auth import require_login
from utils.annotations import load_annotations, apply_annotations, apply_annotations_to_gdf, ensure_human_labeled
from utils.indices import load_predictions_indices, EFFECTIVE_PREDICTIONS_PATH
from utils.styling import inject_landing_css
from utils.taxonomy import species_display, to_scientific

st.set_page_config(
    page_title="Bureau of Ocean Energy Management - Gulf of America Biodiversity Survey",
    page_icon="🦅",
    layout="wide"
)

# Require login for all content
require_login()
inject_landing_css()

# Species name preference: common (default) vs scientific
if "use_common_names" not in st.session_state:
    st.session_state.use_common_names = True

# Add the app directory to Python path
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

data_path = Path(__file__).parent / "data" / "most_recent_all_flight_predictions.csv"
if not data_path.exists():
    st.error(f"File not found: {data_path}")
    st.stop()


@st.cache_data
def _load_effective_predictions():
    if Path(EFFECTIVE_PREDICTIONS_PATH).exists():
        return pd.read_csv(EFFECTIVE_PREDICTIONS_PATH)
    return None


@st.cache_data
def _load_predictions_with_annotations():
    df = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")
    ann = load_annotations("app/data/annotations.csv")
    df = apply_annotations(df, ann, id_col="crop_image_id", label_col="cropmodel_label", set_col="set")
    return ensure_human_labeled(df, set_col="set")


@st.cache_data
def _load_indices():
    return load_predictions_indices()


# Read the data (prefer effective CSV when available)
_effective = _load_effective_predictions()
if _effective is not None:
    df = _effective.copy()
else:
    df = _load_predictions_with_annotations().copy()

df = df.loc[df.score > 0.7]
df = df[df["cropmodel_label"] != "FalsePositive"]
df = df[df["cropmodel_label"].str.count(" ") == 1]

annotations_df = load_annotations("app/data/annotations.csv")
gdf = gpd.read_file(data_path.parent / "all_predictions.shp")
gdf = apply_annotations_to_gdf(gdf, annotations_df, gdf_image_col="crop_image", gdf_label_col="cropmodel_", gdf_set_col="set")
gdf['date'] = pd.to_datetime(gdf['timestamp'], errors='coerce')

st.title("Bureau of Ocean Energy Management Biodiversity Survey")
st.markdown("*Visualize biodiversity from aerial surveys of offshore energy areas—AI-assisted detection and classification of marine wildlife to support environmental assessment.*")

# Toggle: show species as common names (default) or scientific names app-wide
st.session_state.use_common_names = st.toggle(
    "Show species as common names",
    value=st.session_state.use_common_names,
    help="When on, species are shown in English (e.g. Spotted Sandpiper). When off, scientific names are used (e.g. Actitis macularius). This setting applies across the app.",
    key="species_name_toggle",
)
use_common = st.session_state.use_common_names

with st.expander("About this tool"):
    st.markdown("Use this tool to explore biodiversity data from airborne surveys. The pipeline using AI vision models to detect and classify marine species in aerial imagery. The outputs of these models inform offshore project development with rapid, cost-effective biological monitoring. The viewer provides interactive maps, species analysis, and image galleries of detections. The tool was developed by the University of Florida in collaboration with the Bureau of Ocean Energy Management, US Fish and Wildlife Service, and USGS")

# Conceptual figure (hero): highlight workflow
_conceptual_path = app_dir / "www" / "conceptual_figure.png"
if not _conceptual_path.exists():
    _conceptual_path = Path("app/www/conceptual_figure.png")
if _conceptual_path.exists():
    conceptual_figure = Image.open(_conceptual_path)
    st.image(conceptual_figure, caption="**Figure 1.** Raw flight image with detections and classifications overlaid (e.g. bottlenose dolphins).", use_container_width=True)

st.markdown("### Progress")
df_labeled = ensure_human_labeled(df, "set")
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Observations", f"{len(df):,}")
with m2:
    st.metric("Human-reviewed", f"{df_labeled['human_labeled'].sum():,}")
with m3:
    st.metric("Species", df["cropmodel_label"].nunique())
st.caption(f"Aerial surveys: {df['flight_name'].nunique()}")

# Model performance (held-out evaluation data)
st.markdown("### Model performance")
st.caption("Metrics on images not used for training. Recall = proportion of true objects detected; precision = proportion of predictions that are correct.")
det_loaded, cls_loaded = False, False
try:
    detection_metrics_df = pd.read_csv("app/data/detection_model_metrics.csv")
    det_loaded = not detection_metrics_df.empty
except Exception:
    pass
try:
    classification_metrics_df = pd.read_csv("app/data/classification_model_metrics.csv")
    cls_loaded = not classification_metrics_df.empty
except Exception:
    pass
if det_loaded or cls_loaded:
    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        if det_loaded:
            latest = detection_metrics_df.sort_values("timestamp").groupby("metricName").last()
            st.markdown("**Detection**")
            if "box_recall" in latest.index:
                st.metric("Recall", f"{latest.loc['box_recall', 'metricValue'] * 100:.1f}%")
            if "box_precision" in latest.index:
                st.metric("Precision", f"{latest.loc['box_precision', 'metricValue'] * 100:.1f}%")
            if "empty-frame-precision" in latest.index:
                st.caption(f"Empty-frame precision: {latest.loc['empty-frame-precision', 'metricValue'] * 100:.1f}%")
            if "empty_frame_accuracy" in latest.index:
                st.caption(f"Empty-frame accuracy: {latest.loc['empty_frame_accuracy', 'metricValue'] * 100:.1f}%")
            try:
                det_exp = detection_metrics_df.sort_values("timestamp").iloc[-1]["experiment"]
                st.markdown(f'<a href="https://www.comet.com/bw4sz/boem/{det_exp}" target="_blank">View detection experiment in Comet →</a>', unsafe_allow_html=True)
            except Exception:
                pass
    with perf_col2:
        if cls_loaded:
            latest = classification_metrics_df.sort_values("timestamp").groupby("metricName").last()
            st.markdown("**Classification**")
            # Classification experiments log "Micro-Average Accuracy" / "Micro-Average Precision", not "Accuracy"/"Precision"
            if "Micro-Average Accuracy" in latest.index:
                st.metric("Micro-Average Accuracy", f"{latest.loc['Micro-Average Accuracy', 'metricValue'] * 100:.1f}%")
            elif "Accuracy" in latest.index:
                st.metric("Accuracy", f"{latest.loc['Accuracy', 'metricValue'] * 100:.1f}%")
            if "Micro-Average Precision" in latest.index:
                st.metric("Micro-Average Precision", f"{latest.loc['Micro-Average Precision', 'metricValue'] * 100:.1f}%")
            elif "Precision" in latest.index:
                st.metric("Precision", f"{latest.loc['Precision', 'metricValue'] * 100:.1f}%")
            try:
                last_row = classification_metrics_df.sort_values("timestamp").iloc[-1]
                exp_key = last_row.get("experimentKey", None)
                exp_name = last_row["experiment"]
                link_id = exp_key if (exp_key is not None and pd.notna(exp_key) and str(exp_key).strip()) else exp_name
                st.markdown(f'<a href="https://www.comet.com/bw4sz/boem/{link_id}?experiment-tab=confusionMatrix" target="_blank">Confusion matrix →</a>', unsafe_allow_html=True)
            except Exception:
                pass

st.markdown("---")
st.header("Observations")

with st.expander("How to use"):
    st.markdown("""
- Click on any point on the map to view its metadata
- Use the confidence score slider to filter observations
- Select species to show/hide from the map
""")

# Add download button for all predictions
try:
    _dl = _load_effective_predictions()
    predictions_df = _dl if _dl is not None else _load_predictions_with_annotations()
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

m = leafmap.Map(center=[28, -90], zoom=6)
app_data_dir = Path(__file__).parent / "data"
default_file = app_data_dir / "all_predictions.shp"
if default_file.exists():
    gdf_obs = gpd.read_file(default_file)
    # Normalize shapefile column names (ESRI Shapefile truncates to 10 chars; prepare uses Lat/Lon, timestamp)
    _cols = gdf_obs.columns
    if "Lat" in _cols and "lat" not in _cols:
        gdf_obs["lat"] = gdf_obs["Lat"]
    if "Lon" in _cols and "long" not in _cols:
        gdf_obs["long"] = gdf_obs["Lon"]
    if "timestamp" in _cols and "date" not in _cols:
        gdf_obs["date"] = pd.to_datetime(gdf_obs["timestamp"], errors="coerce")
    if "flight_nam" in _cols and "flight_n_1" not in _cols:
        gdf_obs["flight_n_1"] = gdf_obs["flight_nam"]
    elif "flight_name" in _cols and "flight_n_1" not in _cols:
        gdf_obs["flight_n_1"] = gdf_obs["flight_name"]
    if "human_lab" in _cols and "human_labeled" not in _cols:
        gdf_obs["human_labeled"] = gdf_obs["human_lab"]
    if "crop_imag" in _cols and "crop_image" not in _cols:
        gdf_obs["crop_image"] = gdf_obs["crop_imag"]
    elif "crop_image_" in _cols and "crop_image" not in _cols:
        gdf_obs["crop_image"] = gdf_obs["crop_image_"]
    if "cropmodel_l" in _cols and "cropmodel_" not in _cols:
        gdf_obs["cropmodel_"] = gdf_obs["cropmodel_l"]
    gdf_obs = gdf_obs[gdf_obs['cropmodel_'].notna()]
    gdf_obs = apply_annotations_to_gdf(gdf_obs, annotations_df, gdf_image_col="crop_image", gdf_label_col="cropmodel_", gdf_set_col="set")
    gdf_obs = gdf_obs[gdf_obs['cropmodel_'] != "FalsePositive"]
    gdf_obs = ensure_human_labeled(gdf_obs, set_col="set")
    indices = _load_indices()
    if indices:
        unique_labels = indices["species_list"]
    else:
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
        format_func=lambda x: species_display(x, use_common),
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
    # Custom pane so observation markers draw on top of other overlays (e.g. BOEM layers)
    folium.map.CustomPane("observations_top", z_index=450, pointer_events=True).add_to(m    )
    # Custom pane so observation markers draw on top of other overlays (e.g. BOEM layers)
    folium.map.CustomPane("observations_top", z_index=450, pointer_events=True).add_to(m)
    _boem_lease_blocks_url = (
        "https://gis.boem.gov/arcgis/rest/services/BOEM_BSEE/MMC_Layers/MapServer/11/query"
        "?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=geojson"
    )
    # _wind_lease_url = (
    #     "https://services7.arcgis.com/G5Ma95RzqJRPKsWL/arcgis/rest/services/"
    #     "Wind_Lease_Boundaries__BOEM_/FeatureServer/8/query"
    #     "?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=geojson"
    # )
    # _wind_planning_url = (
    #     "https://services7.arcgis.com/G5Ma95RzqJRPKsWL/arcgis/rest/services/"
    #     "Wind_Planning_Areas__BOEM_/FeatureServer/7/query"
    #     "?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=geojson"
    # )
    try:
        m.add_geojson(
            _boem_lease_blocks_url,
            layer_name="BOEM OCS Lease Blocks",
            zoom_to_layer=False,
            info_mode=None,
            style={"color": "#686868", "weight": 0.4, "fillOpacity": 0},
        )
    except Exception:
        pass  # layer optional; service may be unavailable
    # try:
    #     m.add_geojson(
    #         _wind_lease_url,
    #         layer_name="Wind Lease Boundaries",
    #         zoom_to_layer=False,
    #         info_mode="on_hover",
    #         style={"color": "#0066cc", "weight": 2, "fillOpacity": 0.15},
    #     )
    # except Exception:
    #     pass
    # try:
    #     m.add_geojson(
    #         _wind_planning_url,
    #         layer_name="Wind Planning Areas",
    #         zoom_to_layer=False,
    #         info_mode="on_hover",
    #         style={"color": "#2d862d", "weight": 2, "fillOpacity": 0.1},
    #     )
    # except Exception:
    #     pass
    # m.add_layer_control()
    gallery_gdf = None  # set in else when we have filtered data; shown below the map
    try:
        # Ensure we filter by scientific names (selector may show common names)
        selected_scientific = [to_scientific(l) for l in selected_labels]
        filtered_gdf = gdf_obs[
            (gdf_obs['score'] >= score_threshold) &
            (gdf_obs['cropmodel_'].isin(selected_scientific))
        ]
        if human_reviewed:
            filtered_gdf = filtered_gdf[filtered_gdf['human_labeled'] == True]
        if len(filtered_gdf) == 0:
            st.warning("No observations meet the selected filters")
        else:
            # Remove image_link/crop_api_path logic; rename to public-facing "Species"
            filtered_gdf = filtered_gdf[['score', 'cropmodel_', 'cropmode_1', 'set',
                            'flight_n_1', 'date', 'lat', 'long', 'crop_image', 'geometry']].copy()
            filtered_gdf = filtered_gdf.rename(columns={
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
            filtered_gdf['Species'] = filtered_gdf['Species'].map(lambda s: species_display(s, use_common))
            gallery_gdf = filtered_gdf

            m.add_gdf(
                filtered_gdf,
                layer_name=f"Observations (score ≥ {score_threshold})",
                style={'color': "#0000FF"},
                info_mode='on_click',
                hover_style={'sticky': True},
                info_columns=['Image'],
                pane='observations_top',  # draw markers on top of other overlays
            )
    except Exception as e:
        st.error(f"Error processing vector data: {str(e)}")
    m.to_streamlit(height=700, width=None)

    # Selected Observations gallery and download (below the map)
    if gallery_gdf is not None:
        gallery_images = gallery_gdf['Image'].dropna().unique()[:20]
        if len(gallery_images) > 0:
            st.subheader("Selected Observations")
            cols = st.columns(3)
            for idx, img in enumerate(gallery_images):
                img_path = Path("app/data/images") / str(img)
                with cols[idx % 3]:
                    if img_path.exists():
                        row = gallery_gdf[gallery_gdf['Image'] == img].iloc[0]
                        species_name = row['Species']
                        caption = species_name if human_reviewed else f"{species_name} (Confidence: {row['Species Confidence']:.2f})"
                        st.image(str(img_path), caption=caption, use_container_width=True)
                    else:
                        st.info(f"Image {img} not found.")
        else:
            st.info("No images available for the selected observations.")

        try:
            temp_file = app_data_dir / "temp_filtered.shp"
            gallery_gdf.to_file(temp_file)
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

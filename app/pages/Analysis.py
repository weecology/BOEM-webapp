import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import pydeck as pdk
import h3
from utils.styling import load_css
from utils.annotations import load_annotations, apply_annotations, apply_annotations_to_gdf, ensure_human_labeled
import geopandas as gpd
from utils.auth import require_login
from utils.taxonomy import species_display

def create_label_count_plots(label_counts_df, use_common_names=True):
    """Create plots showing label distributions. Species shown as common or scientific name."""
    if label_counts_df is None:
        return None
    counts_df = label_counts_df.groupby('cropmodel_label').size().reset_index(name='count')
    counts_df.columns = ['label', 'count']
    counts_df = counts_df[counts_df['label'] != '0']
    counts_df['Species'] = counts_df['label'].map(lambda s: species_display(s, use_common_names))
    label_order = counts_df.sort_values('count', ascending=False)['Species']
    fig_hist = px.bar(counts_df,
                      x='Species',
                      y='count',
                      title='Species Abundance Across All Surveys',
                      labels={
                          'Species': 'Species',
                          'count': 'Number of Instances',
                      },
                      category_orders={"Species": label_order.tolist()})
    return fig_hist


def app():
    require_login()
    use_common = st.session_state.get("use_common_names", True)
    st.title("Species Analysis")

    # About this page
    st.markdown("""
    This page provides an analysis of the species abundance across all surveys. It includes a plot of the species abundance across all surveys, a table of the species abundance across all surveys, and a plot of the rare species across all surveys. This page differentiates, 'predicted' observations from the model versus human-reviewd obserations that have been verified.
    """)

    app_data_dir = Path(__file__).parents[1] / "data"
    default_file = app_data_dir / "most_recent_all_flight_predictions.csv"
    df = pd.read_csv(default_file)

    # Apply annotation overrides to analysis dataset
    annotations_df = load_annotations("app/data/annotations.csv")
    df = apply_annotations(df,
                           annotations_df,
                           id_col="crop_image_id",
                           label_col="cropmodel_label",
                           set_col="set")
    df = ensure_human_labeled(df, set_col="set")
    df = df[df["cropmodel_label"] != "FalsePositive"]
    df = df[df["cropmodel_label"].str.count(" ") == 1]

    # Score filters: human-reviewed observations are always included
    min_detection = st.slider(
        "Min detection score",
        min_value=0.0,
        max_value=1.0,
        value=0.92,
        step=0.05,
        help="Minimum detection confidence (0–1). Human-reviewed observations are always included.",
    )
    min_classification = st.slider(
        "Min classification score",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Minimum species classification confidence (0–1). Human-reviewed observations are always included.",
    )
    # Include human-reviewed always; otherwise require both scores above thresholds
    if "cropmodel_score" in df.columns:
        score_ok = df["human_labeled"] | (
            (df["score"] >= min_detection) & (df["cropmodel_score"] >= min_classification)
        )
    else:
        score_ok = df["human_labeled"] | (df["score"] >= min_detection)
    df = df[score_ok]

    date_component = df["flight_name"].str.split("_").str[1]
    datetime_values = pd.to_datetime(date_component,
                                     format='%Y%m%d',
                                     errors='coerce')
    df["timestamp"] = datetime_values.astype(str)
    gdf = gpd.read_file(default_file.parent / "all_predictions.shp")
    gdf['date'] = pd.to_datetime(gdf['timestamp'], errors='coerce')

    # Normalize shapefile column names (ESRI truncates to 10 chars)
    _cols = gdf.columns
    if "Lat" in _cols and "lat" not in _cols:
        gdf["lat"] = gdf["Lat"]
    if "Lon" in _cols and "long" not in _cols:
        gdf["long"] = gdf["Lon"]
    if "crop_imag" in _cols and "crop_image" not in _cols:
        gdf["crop_image"] = gdf["crop_imag"]
    elif "crop_image_" in _cols and "crop_image" not in _cols:
        gdf["crop_image"] = gdf["crop_image_"]
    if "cropmodel_l" in _cols and "cropmodel_" not in _cols:
        gdf["cropmodel_"] = gdf["cropmodel_l"]
    if "human_lab" in _cols and "human_labeled" not in _cols:
        gdf["human_labeled"] = gdf["human_lab"]
    if "cropmode_1" in _cols and "cropmodel_score" not in _cols:
        gdf["cropmodel_score"] = gdf["cropmode_1"]
    gdf = gdf[gdf["cropmodel_"].notna()]
    gdf = apply_annotations_to_gdf(
        gdf, annotations_df,
        gdf_image_col="crop_image",
        gdf_label_col="cropmodel_",
        gdf_set_col="set",
    )
    gdf = gdf[gdf["cropmodel_"] != "FalsePositive"]
    gdf = gdf[gdf["cropmodel_"].str.count(" ") == 1]
    gdf = ensure_human_labeled(gdf, set_col="set")
    if "lat" not in gdf.columns and gdf.geometry is not None:
        gdf["lat"] = gdf.geometry.y
        gdf["long"] = gdf.geometry.x
    # Apply same score filters as df (human-reviewed always included)
    if "score" in gdf.columns:
        if "cropmodel_score" in gdf.columns:
            gdf_score_ok = gdf["human_labeled"] | (
                (gdf["score"] >= min_detection) & (gdf["cropmodel_score"] >= min_classification)
            )
        else:
            gdf_score_ok = gdf["human_labeled"] | (gdf["score"] >= min_detection)
        gdf = gdf[gdf_score_ok]

    # Hexbin maps: species richness and detection abundance
    st.subheader("Spatial patterns across all surveys")
    st.markdown("Hexbin maps show **species richness** (number of distinct species) and **detection abundance** (number of detections) per hex cell over the survey area.")
    if gdf.empty or "lat" not in gdf.columns:
        st.info("No georeferenced detection data available for mapping.")
    else:
        try:
            # Coerce to numeric and keep only valid WGS84 coordinates (h3 fails on NaN/invalid)
            gdf_hex = gdf.copy()
            gdf_hex["lat"] = pd.to_numeric(gdf_hex["lat"], errors="coerce")
            gdf_hex["long"] = pd.to_numeric(gdf_hex["long"], errors="coerce")
            gdf_hex = gdf_hex.dropna(subset=["lat", "long"])
            gdf_hex = gdf_hex[
                (gdf_hex["lat"] >= -90) & (gdf_hex["lat"] <= 90)
                & (gdf_hex["long"] >= -180) & (gdf_hex["long"] <= 180)
            ]
            if gdf_hex.empty:
                st.info("No valid coordinates for hexbin mapping.")
            else:
                h3_res = 5
                gdf_hex["hex"] = gdf_hex.apply(
                    lambda r: h3.latlng_to_cell(float(r["lat"]), float(r["long"]), h3_res),
                    axis=1,
                )
                hex_agg = (
                    gdf_hex.groupby("hex")
                    .agg(
                        abundance=("cropmodel_", "size"),
                        richness=("cropmodel_", "nunique"),
                    )
                    .reset_index()
                )
                if hex_agg.empty:
                    st.info("No detections could be aggregated into hexbins.")
                else:
                    # Scale for color: 0–255
                    ab = hex_agg["abundance"]
                    rn = hex_agg["richness"]
                    hex_agg["count"] = ab
                    hex_agg["abun_255"] = (
                        (ab - ab.min()) / (ab.max() - ab.min() + 1e-9) * 255
                    ).astype(int)
                    hex_agg["rich_255"] = (
                        (rn - rn.min()) / (rn.max() - rn.min() + 1e-9) * 255
                    ).astype(int)
                    view = pdk.ViewState(
                        latitude=28.0,
                        longitude=-90.0,
                        zoom=5,
                        pitch=0,
                        bearing=0,
                    )
                    # Abundance map
                    layer_abun = pdk.Layer(
                        "H3HexagonLayer",
                        hex_agg,
                        pickable=True,
                        stroked=True,
                        filled=True,
                        extruded=False,
                        get_hexagon="hex",
                        get_fill_color="[255 - abun_255, 220, abun_255]",
                        get_line_color=[255, 255, 255],
                        line_width_min_pixels=1,
                    )
                    st.markdown("**Detection abundance** (number of detections per hex)")
                    st.pydeck_chart(
                        pdk.Deck(
                            layers=[layer_abun],
                            initial_view_state=view,
                            map_style="light",
                            tooltip={"text": "Detections: {count}"},
                        )
                    )
                    # Richness map
                    layer_rich = pdk.Layer(
                        "H3HexagonLayer",
                        hex_agg,
                        pickable=True,
                        stroked=True,
                        filled=True,
                        extruded=False,
                        get_hexagon="hex",
                        get_fill_color="[100, 255 - rich_255, rich_255]",
                        get_line_color=[255, 255, 255],
                        line_width_min_pixels=1,
                    )
                    st.markdown("**Species richness** (number of distinct species per hex)")
                    st.pydeck_chart(
                        pdk.Deck(
                            layers=[layer_rich],
                            initial_view_state=view,
                            map_style="light",
                            tooltip={"text": "Species: {richness}"},
                        )
                    )
        except Exception as e:
            st.error("Could not build hexbin maps.")
            st.exception(e)

    st.subheader("Predicted Species")
    hist_plot = create_label_count_plots(df, use_common)
    st.plotly_chart(hist_plot, use_container_width=True)
    species_table = df.groupby('cropmodel_label').size().sort_values(
        ascending=False).reset_index(name='count')
    species_table = species_table.rename(
        columns={'cropmodel_label': 'Species'})
    species_table['Species'] = species_table['Species'].map(
        lambda s: species_display(s, use_common))
    st.write(species_table)

    # Rare species plot
    species_counts = df["cropmodel_label"].value_counts()
    if not species_counts.empty:
        max_count = species_counts.iloc[0]
        rare_threshold = max_count * 0.10
        rare_species = species_counts[species_counts < rare_threshold]
        if not rare_species.empty:
            rare_df = df[df['cropmodel_label'].isin(rare_species.index)]
            rare_counts = rare_df['cropmodel_label'].value_counts(
            ).reset_index()
            rare_counts.columns = ['label', 'count']
            rare_counts['Species'] = rare_counts['label'].map(
                lambda s: species_display(s, use_common))
            rare_fig = px.bar(
                rare_counts,
                x='Species',
                y='count',
                title='Predicted Rare Species (<10% of Most Common)',
                labels={
                    'Species': 'Species',
                    'count': 'Number of Instances'
                })
            st.plotly_chart(rare_fig, use_container_width=True)
        else:
            st.info(
                'No rare species (less than 10% of the most common) found in this dataset.'
            )
    else:
        st.info('No species data available.')

    st.subheader("Reviewed Observations")
    st.markdown("""
    Our human-in-loop system recommends which images to review for accuracy based on model confidence scores."
    """)
    reviewed_species_counts = df[df['human_labeled'] ==
                                 True]['cropmodel_label'].value_counts()
    if not reviewed_species_counts.empty:
        reviewed_counts = reviewed_species_counts.reset_index()
        reviewed_counts.columns = ['label', 'count']
        reviewed_counts['Species'] = reviewed_counts['label'].map(
            lambda s: species_display(s, use_common))
        reviewed_fig = px.bar(reviewed_counts,
                              x='Species',
                              y='count',
                              labels={
                                  'Species': 'Species',
                                  'count': 'Number of Instances'
                              })
        st.plotly_chart(reviewed_fig, use_container_width=True)
        max_count = reviewed_species_counts.iloc[0]
        rare_threshold = max_count * 0.10
        rare_species = reviewed_species_counts[reviewed_species_counts <
                                               rare_threshold]
        if not rare_species.empty:
            # Show counts of human-reviewed instances only (not total predictions)
            reviewed_df = df[df['human_labeled'] == True]
            rare_df = reviewed_df[reviewed_df['cropmodel_label'].isin(
                rare_species.index)]
            rare_counts = rare_df['cropmodel_label'].value_counts(
            ).reset_index()
            rare_counts.columns = ['label', 'count']
            rare_counts['Species'] = rare_counts['label'].map(
                lambda s: species_display(s, use_common))
            rare_fig = px.bar(
                rare_counts,
                x='Species',
                y='count',
                title=
                'Rare Species by Human Review (<10% of Most Common Reviewed Species)',
                labels={
                    'Species': 'Species',
                    'count': 'Human-Reviewed Count'
                })
            st.plotly_chart(rare_fig, use_container_width=True)
        else:
            st.info(
                'No rare species (less than 10% of the most common) found in this dataset.'
            )
    else:
        st.info('No species data available.')

if __name__ == "__main__":
    load_css()
    app()

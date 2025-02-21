import streamlit as st
import leafmap.foliumap as leafmap
from pathlib import Path
import os
from utils.styling import load_css

# Must be the first Streamlit command
st.set_page_config(
    page_title="Biodiversity Heatmap",
    page_icon="🗺️",
    layout="wide"
)

def app():
    st.title("Biodiversity Heatmap")

    # Get the app's data directory
    app_data_dir = Path(__file__).parents[1] / "data"
    heatmaps_dir = app_data_dir / "heatmaps"

    if not heatmaps_dir.exists():
        st.error("No heatmap data found. Please run prepare.py first to generate heatmaps.")
        return

    # Get list of heatmap files
    heatmap_files = list(heatmaps_dir.glob("*_heatmap.tif"))
    
    if not heatmap_files:
        st.error("No heatmap files found in the heatmaps directory.")
        return

    # Initialize map
    m = leafmap.Map()
    m.add_basemap("OpenStreetMap")

    # Add each heatmap to the map with a colormap
    for heatmap_file in heatmap_files:
        flight_name = heatmap_file.stem.replace('_heatmap', '')
        
        try:
            m.add_raster(
                str(heatmap_file),
                layer_name=f"Heatmap - {flight_name}",
                colormap='viridis',  # You can change this to other colormaps like 'hot', 'YlOrRd', etc.
                opacity=0.7
            )
        except Exception as e:
            st.error(f"Error loading heatmap for {flight_name}: {str(e)}")

    # Display the map
    m.to_streamlit(height=700)

    # Add description
    st.info("""
    **About this map:**
    - The heatmap shows the density of wildlife observations across different flight lines
    - Darker colors indicate higher concentration of observations
    - Each flight line's data is shown as a separate layer
    """)

if __name__ == "__main__":
    load_css()
    app()
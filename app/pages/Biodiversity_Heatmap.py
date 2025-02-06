import streamlit as st
import leafmap.foliumap as leafmap
from utils.raster_utils import convert_to_cog, create_pyramid_tiles
import os
import asyncio
import nest_asyncio
import threading
from pathlib import Path

# Initialize async event loop
nest_asyncio.apply()

st.set_page_config(
    page_title="Raster Analysis",
    page_icon="📊",
    layout="wide"
)

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        raise

@st.cache_data
def load_raster(file_path, max_size_mb=200):
    """Load and optimize large raster for web viewing"""
    try:
        # Create output paths
        cog_path = file_path.replace('.tif', '_cog.tif')
        tiles_dir = file_path.replace('.tif', '_tiles')
        
        # Convert to COG if needed
        if not os.path.exists(cog_path):
            convert_to_cog(file_path, cog_path, max_size_mb)
        
        # Create pyramid tiles if needed
        if not os.path.exists(tiles_dir):
            create_pyramid_tiles(cog_path, tiles_dir)
        
        return cog_path, tiles_dir
        
    except Exception as e:
        st.error(f"Error processing raster: {str(e)}")
        return None, None

def app():
    st.title("Raster Data Viewer")

    # Get the app's data directory
    app_data_dir = Path(__file__).parents[1] / "data"

    # Add base map toggle in sidebar
    show_basemap = st.sidebar.checkbox("Show OpenStreetMap", value=False)

    # Initialize map
    m = leafmap.Map(center=[20, 0], zoom=2)
    if show_basemap:
        m.add_basemap("OpenStreetMap")

    # Add option to select data source
    data_source = st.radio(
        "Select Data Source",
        ["Default Data Directory", "External Path"]
    )

    if data_source == "Default Data Directory":
        # Scan data directory for raster files
        raster_files = list(app_data_dir.rglob("*.tif"))
        
        if raster_files:
            # Create relative paths for display
            rel_paths = [str(f.relative_to(app_data_dir)) for f in raster_files]
            selected_file = st.selectbox(
                "Select Raster File",
                rel_paths,
                format_func=lambda x: x
            )
            if selected_file:
                input_file = app_data_dir / selected_file
            else:
                input_file = None
        else:
            st.warning("No raster files found in data directory")
            input_file = None

    else:  # External Path
        file_path = st.text_input(
            "Enter full path to raster file",
            help="Full path to .tif file"
        )
        input_file = Path(file_path) if file_path else None

    if input_file:
        try:
            if not input_file.exists():
                st.error(f"File not found: {input_file}")
            else:
                # Process and display raster
                raster_path, tiles_dir = load_raster(str(input_file))
                
                # Add raster to map
                m.add_raster(
                    raster_path,
                    layer_name="Raster",
                    fit_bounds=True
                )
                
                # Add download button for processed file if available
                if raster_path and Path(raster_path).exists():
                    with open(raster_path, 'rb') as f:
                        st.download_button(
                            label="Download processed file",
                            data=f.read(),
                            file_name=f"processed_{input_file.name}",
                            mime="image/tiff"
                        )
                
        except Exception as e:
            st.error(f"Error processing raster: {str(e)}")
    
    m.to_streamlit(height=600)

if __name__ == "__main__":
    app()
# IMPORTANT: This must be the first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="Geospatial Data Viewer",
    page_icon="🌍",
    layout="wide"
)

import sys
from pathlib import Path
import leafmap.foliumap as leafmap

# Add the app directory to Python path
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

# Create the sidebar navigation with "Main" instead of "About"
page = st.sidebar.selectbox(
    "Navigation", 
    ["Main", "Raster Viewer", "Vector Viewer", "WebMap"]
)

# Display the selected page
if page == "Main":
    st.title("Geospatial Data Viewer")
    
    st.markdown("""
    ## Welcome to the Geospatial Data Viewer!
    
    This application provides interactive visualization and processing of geospatial data:
    
    ### Features:
    - **Raster Viewer**: View and analyze raster datasets (GeoTIFF)
    - **Vector Viewer**: Explore vector data with interactive styling
    - **WebMap**: View MBTiles and web mapping services
    
    ### Supported Formats:
    - Raster: GeoTIFF (.tif, .tiff)
    - Vector: Shapefile (.shp), GeoJSON (.geojson)
    - Tiles: MBTiles (.mbtiles)
    
    ### Getting Started:
    1. Select a viewer from the sidebar
    2. Upload your data
    3. Interact with the map
    
    ### Data Processing:
    - Automatic optimization for large files
    - Cloud-Optimized GeoTIFF (COG) conversion
    - Vector tiling and simplification
    """)
    
    # Display a sample map
    m = leafmap.Map(center=[0, 0], zoom=2)
    m.add_basemap("OpenStreetMap")
    m.to_streamlit(height=500)

elif page == "Raster Viewer":
    from pages.raster_viewer import app
    app()
elif page == "Vector Viewer":
    from pages.vector_viewer import app
    app()
elif page == "WebMap":
    from pages.webmap import app
    app()

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
    st.title("Project Overview")

    st.markdown("""

    Text about the project and the goals of the project
    
    ### Current Status

    Graphs and tables of the current status of the project

    ### Future Work

    Future work and ideas for the project

    """)

elif page == "Raster Viewer":
    from pages.raster_viewer import app
    app()
elif page == "Vector Viewer":
    from pages.vector_viewer import app
    app()
elif page == "WebMap":
    from pages.webmap import app
    app()

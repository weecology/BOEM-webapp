import streamlit as st
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

import sys
from pathlib import Path
import leafmap.foliumap as leafmap

# Add the app directory to Python path
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

st.title("BOEM Offshore Biodiversity Surveys")

st.markdown("""
    Welcome to the BOEM Offshore Biodiversity Survey Data Viewer. 
    This application provides tools for visualizing and analyzing biodiversity data collected during aerial surveys of offshore wind energy areas.

    ### Project Overview
    
    This project processes aerial survey data to:
    - Detect and classify marine wildlife species
    - Generate distribution maps and abundance estimates
    - Analyze temporal and spatial patterns
    - Support environmental impact assessments

    The data viewer includes:
    - Interactive maps for viewing survey tracks and observations
    - Analysis tools for exploring species distributions
    - Video playback of flight line footage
    - Image galleries of detected species
    
    For detailed information on model development and performance metrics, visit our [CometML Dashboard](https://www.comet.com/bw4sz/).

    """)
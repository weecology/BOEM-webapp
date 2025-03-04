import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
from utils.styling import load_css

st.set_page_config(
    page_title="BOEM Wildlife Detector",
    page_icon="🦅",
    layout="wide"
)

import sys
from pathlib import Path
import leafmap.foliumap as leafmap
import pandas as pd

# Add the app directory to Python path
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

st.title("BOEM Wildlife Detector")

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
    - Image galleries of detected species
    
    For detailed information on model development and performance metrics, visit our [CometML Dashboard](https://www.comet.com/bw4sz/).
""")

# Read the data
data_path = Path(__file__).parent / "data" / "most_recent_all_flight_predictions.csv"

if not data_path.exists():
    st.error(f"File not found: {data_path}")
    st.stop()

df = pd.read_csv(data_path)

# Convert date with specific format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Display basic statistics
st.header("Dataset Overview")
st.write(f"Current Flight: {df['flight_name'].unique()[0]}")
st.write(f"Total Records: {len(df)}")
st.write(f"Number of Species: {df['label'].nunique()}")
st.write(f"Total Sites: {df['flight_name'].nunique()}")
st.write(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

# Top N species by count
n_species = 10

# Create species count plot
species_counts = df.groupby('label').size().sort_values(ascending=False).head(n_species)

fig = px.bar(
    x=species_counts.index,
    y=species_counts.values,
    title=f"Top {n_species} Most Common Bird Species",
    labels={'x': 'Species', 'y': 'Total Count'}
)
fig.update_layout(
    xaxis_tickangle=-45,
    height=500,
    width=800
)
st.plotly_chart(fig)
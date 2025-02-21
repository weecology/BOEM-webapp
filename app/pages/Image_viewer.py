import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from utils.styling import load_css

def app():
    st.title("Image Viewer")

    # Read the data to get species list
    data_path = Path("app/data/predictions.csv")

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['label'])
    species_list = df['label'].unique()

    # Create image directory path
    image_dir = Path("app/data/images")

    # Initialize session state if needed
    if 'selected_species' not in st.session_state:
        st.session_state.selected_species = "Bird"

    # Use standard Streamlit selector with USWDS styling
    selected_species = st.selectbox(
        "Select a species",
        options=species_list,
        index=0
    )

    # Update session state
    st.session_state.selected_species = selected_species

    # Add score filter slider
    min_score = st.slider(
        "Filter by minimum confidence score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05
    )

    # Get image paths for the selected species from predictions.csv and filter by score
    st.header("Results")
    species_df = df[df['label'] == st.session_state.selected_species]
    filtered_df = species_df[species_df['score'] >= min_score]
    image_paths = filtered_df['image_path'].unique()

    # Check which images exist in the image directory
    existing_images = [os.path.join(image_dir, image_path) for image_path in image_paths if image_dir / image_path in image_dir.glob('*.jpg')]

    # Display the images as a gallery
    # Caption with the prediction score
    for image_path in existing_images:
        score = df[df['image_path'] == os.path.basename(image_path)]['score'].values[0]
        st.image(image_path, caption=f"Model Confidence: {score:.2f}")

if __name__ == "__main__":
    load_css()
    app()

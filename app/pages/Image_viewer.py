import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
def app():
    st.title("Image Viewer")

    # Read the data to get species list
    data_path = Path("app/data/predictions.csv")

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['label'])
    species_list = df['label'].unique()

    # Create image directory path
    image_dir = Path("app/data/images")

    # Species selection dropdown
    selected_species = st.selectbox(
        "Select a species to view",
        species_list
    )

    # Get image paths for the selected species from predictions.csv
    image_paths = df[df['label'] == selected_species]['image_path'].unique()

    # Check which images exist in the image directory
    existing_images = [os.path.join(image_dir, image_path) for image_path in image_paths if image_dir / image_path in image_dir.glob('*.jpg')]

    # Display the images as a gallery
    # Caption with the prediction score
    for image_path in existing_images:
        score = df[df['image_path'] == os.path.basename(image_path)]['score'].values[0]
        st.image(image_path, caption=f"Model Confidence: {score:.2f}")

if __name__ == "__main__":
    app()

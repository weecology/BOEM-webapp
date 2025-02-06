import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def app():
    st.title("Image Viewer")

    # Read the data to get species list
    data_path = Path("app/data/predictions.csv")
    df = pd.read_csv(data_path)
    species_list = sorted(df['label'].unique())

    # Create image directory path
    image_dir = Path("app/data/images")

    # Species selection dropdown
    selected_species = st.selectbox(
        "Select a species to view",
        species_list
    )

    # Format species name for filename matching
    species_filename = selected_species.lower().replace(' ', '_')
    
    # Look for images with common extensions
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_found = False
    
    for ext in image_extensions:
        image_path = image_dir / f"{species_filename}{ext}"
        if image_path.exists():
            # Display the image
            st.image(str(image_path), caption=selected_species)
            image_found = True
            break
    
    if not image_found:
        st.warning(f"No image found for {selected_species}")

if __name__ == "__main__":
    app()

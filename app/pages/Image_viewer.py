import streamlit as st
from pathlib import Path
from PIL import Image
from utils.styling import load_css
import pandas as pd

def app():
    st.title("Model Prediction Viewer")
    st.text("This page shows validation images and predictions for the latest model")
    # Get experiment data
    image_df = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")
    image_df = image_df[image_df['set'] == 'validation']
    if image_df is None:
        st.warning("No images found in experiments")
        return
        
    # Get unique species
    species_list = sorted(image_df['cropmodel_label'].unique())
    
    # Use standard Streamlit selector with USWDS styling
    selected_species = st.selectbox(
        "Select a species",
        options=species_list,
        index=0
    )
    
    # Filter images by selected species
    species_images = image_df[image_df['cropmodel_label'] == selected_species]
    
    # Limit to 20 images per label
   #species_images = species_images.groupby('cropmodel_label').head(20)
    
    # Create image grid
    cols = st.columns(4)
    for idx, (_, row) in enumerate(species_images.iterrows()):
        with cols[idx % 4]:
            try:
                image = Image.open(f"app/data/images/{row['crop_image_id']}")
                st.image(image)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                    
if __name__ == "__main__":
    load_css()
    app()

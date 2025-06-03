import streamlit as st
from pathlib import Path
from PIL import Image
from utils.styling import load_css
import pandas as pd

def app():
    st.title("Predictions Viewer")
    # Get experiment data
    image_df = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")

    # Detection score slider
    detection_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01
    )

    # Add human-reviewed filter
    human_reviewed = st.checkbox(
        "Human-reviewed",
        value=False,
        help="Show only images that have been reviewed by a human"
    )

    if image_df is None:
        st.warning("No images found in experiments")
        return
        
    # Get unique species
    species_list = sorted(image_df['cropmodel_label'].unique())
    
    # Use standard Streamlit selector with USWDS styling
    # The default index is the most common species
    default_index = species_list.index(image_df.cropmodel_label.value_counts().index[0])
    selected_species = st.selectbox(
        "Select a species",
        options=species_list,
        index=default_index
    )
    
    # Filter images by selected species and detection score and set
    species_images = image_df[(image_df['cropmodel_label'] == selected_species) & (image_df['score'] >= detection_threshold)]
    if human_reviewed:
        species_images = species_images[species_images['set'].isin(['train', 'validation', "review"])]
    
    # Create image grid with 3 columns
    cols = st.columns(3)
    for idx, (_, row) in enumerate(species_images.iterrows()):
        with cols[idx % 3]:
            try:
                image = Image.open(f"app/data/images/{row['crop_image_id']}")
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                    
if __name__ == "__main__":
    load_css()
    app()

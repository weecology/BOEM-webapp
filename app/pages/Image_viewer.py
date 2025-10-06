import streamlit as st
from pathlib import Path
from PIL import Image
from utils.styling import load_css
import pandas as pd
from utils.auth import require_login

def app():
    require_login()
    st.title("Predictions Viewer")
    # Get experiment data
    image_df = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")

    # Convert cropmodel_label to string type
    image_df['cropmodel_label'] = image_df['cropmodel_label'].astype(str)

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
    # Track which image is selected to show caption on click
    if 'selected_image_name' not in st.session_state:
        st.session_state.selected_image_name = None

    for idx, (_, row) in enumerate(species_images.iterrows()):
        with cols[idx % 3]:
            try:
                image_path = f"app/data/images/{row['crop_image_id']}"
                image = Image.open(image_path)
                # Show caption only if this image is selected
                caption_text = row['crop_image_id'] if st.session_state.selected_image_name == row['crop_image_id'] else None
                st.image(image, use_container_width=True, caption=caption_text)
                if st.button("Select", key=f"select_{idx}_{row['crop_image_id']}"):
                    st.session_state.selected_image_name = row['crop_image_id']
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

    # Larger preview of the selected image with filename caption
    if st.session_state.selected_image_name:
        st.subheader("Selected Image")
        preview_path = f"app/data/images/{st.session_state.selected_image_name}"
        try:
            st.image(preview_path, caption=st.session_state.selected_image_name, use_container_width=True)
        except Exception:
            st.info(f"Selected image not found: {st.session_state.selected_image_name}")
                    
if __name__ == "__main__":
    load_css()
    app()

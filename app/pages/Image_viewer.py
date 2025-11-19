import streamlit as st
from pathlib import Path
from PIL import Image
from utils.styling import load_css
import pandas as pd
from utils.auth import require_login
from utils.annotations import load_annotations, apply_annotations


def app():
    require_login()
    st.title("Predictions Viewer")
    # Get experiment data
    image_df = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")
    
    # Apply annotation overrides (label and set)
    annotations_df = load_annotations("app/data/annotations.csv")
    image_df = apply_annotations(image_df, annotations_df, id_col="crop_image_id", label_col="cropmodel_label", set_col="set")

    # Convert cropmodel_label to string type
    image_df['cropmodel_label'] = image_df['cropmodel_label'].astype(str)

    # Detection score slider
    detection_threshold = st.slider("Detection Confidence Threshold",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.8,
                                    step=0.01)

    # Add human-reviewed filter
    human_reviewed = st.checkbox(
        "Human-reviewed",
        value=True,
        help="Show only images that have been reviewed by a human")

    if image_df is None:
        st.warning("No images found in experiments")
        return

    # Build species list based on current human-reviewed filter
    filtered_df = image_df[image_df['set'].isin(
        ['train', 'validation', 'review'])] if human_reviewed else image_df

    if filtered_df.empty:
        st.warning("No records match the current filters.")
        return

    species_list = sorted(filtered_df['cropmodel_label'].unique())

    # Use standard Streamlit selector
    # The default index is the most common species in the filtered data
    default_species = filtered_df['cropmodel_label'].value_counts().index[0]
    default_index = species_list.index(
        default_species) if default_species in species_list else 0
    selected_species = st.selectbox("Select a species",
                                    options=species_list,
                                    index=default_index)

    # Filter images by selected species and detection score against the same filtered set
    species_images = filtered_df[
        (filtered_df['cropmodel_label'] == selected_species)
        & (filtered_df['score'] >= detection_threshold)]

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
                caption_text = row[
                    'crop_image_id'] if st.session_state.selected_image_name == row[
                        'crop_image_id'] else None
                st.image(image, use_container_width=True, caption=caption_text)
                if st.button("Select",
                             key=f"select_{idx}_{row['crop_image_id']}"):
                    st.session_state.selected_image_name = row['crop_image_id']
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

    # Larger preview of the selected image with filename caption
    if st.session_state.selected_image_name:
        st.subheader("Selected Image")
        preview_path = f"app/data/images/{st.session_state.selected_image_name}"
        try:
            st.image(preview_path,
                     caption=st.session_state.selected_image_name,
                     use_container_width=True)
        except Exception:
            st.info(
                f"Selected image not found: {st.session_state.selected_image_name}"
            )

if __name__ == "__main__":
    load_css()
    app()

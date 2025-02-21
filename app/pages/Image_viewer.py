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

    # Initialize session state if needed
    if 'selected_species' not in st.session_state:
        st.session_state.selected_species = "Bird"

    # Add USWDS CSS styling
    st.markdown("""
        <link rel="stylesheet" href="static/css/uswds.min.css">
        <style>
            /* Select styling */
            .stSelectbox > div > div {
                font-family: Source Sans Pro Web, Helvetica Neue, Helvetica, Roboto, Arial, sans-serif;
                font-size: 1.06rem;
                line-height: 1;
                padding: 0.5rem;
                border: 2px solid #565c65;
                border-radius: 0.25rem;
                appearance: none;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 16L4 8h16z' fill='%23565c65'/%3E%3C/svg%3E");
                background-repeat: no-repeat;
                background-position: right 0.75rem center;
                background-size: 0.75rem;
                white-space: nowrap;
                overflow: visible;
                text-overflow: clip;
                height: auto;
                min-height: 2.5rem;
            }
            
            .stSelectbox > div > div:hover {
                border-color: #005ea2;
            }
            
            .stSelectbox > div > div:focus {
                outline: 0.25rem solid #2491ff;
                outline-offset: 0;
            }
            
            .stSelectbox label {
                font-family: Source Sans Pro Web, Helvetica Neue, Helvetica, Roboto, Arial, sans-serif;
                font-size: 1.06rem;
                line-height: 1.1;
                margin-bottom: 0.5rem;
            }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # Use standard Streamlit selector with USWDS styling
    selected_species = st.selectbox(
        "Select a species",
        options=species_list,
        index=0
    )

    # Update session state
    st.session_state.selected_species = selected_species

    # Add score filter slider with USWDS-inspired styling
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
    app()

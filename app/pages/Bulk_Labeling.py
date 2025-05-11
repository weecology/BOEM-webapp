import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import os
from utils.styling import load_css

def app():
    st.title("Bulk Image Labeling")
    st.text("Select multiple images and update their labels in bulk")

    # Load the predictions dataframe
    predictions_df = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")
    
    # Get all unique labels
    all_labels = sorted(predictions_df['cropmodel_label'].unique())
    
    # Create a filter for labels
    selected_labels = st.multiselect(
        "Filter by current labels",
        options=all_labels,
        default=[]
    )
    
    # Filter the dataframe based on selected labels
    if selected_labels:
        filtered_df = predictions_df[predictions_df['cropmodel_label'].isin(selected_labels)]
    else:
        filtered_df = predictions_df
    
    # Get all images from the images directory
    image_dir = Path("app/data/images")
    image_files = list(image_dir.glob("*"))
    
    # Filter images that exist in our dataframe
    valid_images = [img for img in image_files if img.name in filtered_df['crop_image_id'].values]
    
    # Create a grid of selectable images
    st.subheader("Select Images to Relabel")
    
    # Create columns for the image grid
    cols = st.columns(4)
    
    # Store selected images in session state if not already present
    if 'selected_images' not in st.session_state:
        st.session_state.selected_images = set()
    
    # Display images in a grid with checkboxes
    for idx, img_path in enumerate(valid_images):
        with cols[idx % 4]:
            try:
                image = Image.open(img_path)
                st.image(image, use_container_width=True)
                
                # Add checkbox for selection
                if st.checkbox(f"Select {img_path.name}", key=f"select_{img_path.name}"):
                    st.session_state.selected_images.add(img_path.name)
                else:
                    st.session_state.selected_images.discard(img_path.name)
                    
            except Exception as e:
                st.error(f"Error loading image {img_path.name}: {str(e)}")
    
    # New label selection
    st.subheader("Update Labels")
    new_label = st.selectbox(
        "Select new label for selected images",
        options=all_labels
    )
    
    # Update button
    if st.button("Update Labels"):
        if not st.session_state.selected_images:
            st.warning("Please select at least one image")
        else:
            # Update the labels in the dataframe
            for img_id in st.session_state.selected_images:
                predictions_df.loc[predictions_df['crop_image_id'] == img_id, 'cropmodel_label'] = new_label
            
            # Save the updated dataframe
            predictions_df.to_csv("app/data/most_recent_all_flight_predictions.csv", index=False)
            st.success(f"Updated {len(st.session_state.selected_images)} images to label: {new_label}")
            
            # Clear selections
            st.session_state.selected_images.clear()
            st.experimental_rerun()

if __name__ == "__main__":
    load_css()
    app() 
import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import os
import json
from datetime import datetime
from utils.styling import load_css

def get_all_species(taxonomy_data):
    """Extract all species from the taxonomy data"""
    species_list = []
    
    def extract_species(node):
        if node['rank'] == 'Species':
            species_list.append({
                'title': node['title'],
                'scientificName': node['scientificName']
            })
        for child in node.get('children', []):
            extract_species(child)
    
    for node in taxonomy_data:
        extract_species(node)
    
    return species_list

def load_or_create_annotations():
    """Load existing annotations or create new annotations file"""
    annotations_path = Path("app/data/annotations.csv")
    
    if annotations_path.exists():
        return pd.read_csv(annotations_path)
    else:
        # Create new annotations dataframe with required columns
        return pd.DataFrame(columns=[
            'image_id',
            'original_label',
            'new_label',
            'timestamp',
            'user'
        ])

def app():
    st.title("Bulk Image Labeling")
    st.text("Select multiple images and update their labels in bulk")

    # Confidence score slider
    confidence_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01
    )

    # Load the predictions dataframe
    predictions_df = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")
    
    # Filter by confidence score
    if 'score' in predictions_df.columns:
        predictions_df = predictions_df[predictions_df['score'] >= confidence_threshold]
    
    # Load or create annotations dataframe
    annotations_df = load_or_create_annotations()
    
    # Load taxonomy data
    with open("app/data/taxonomy.json", 'r') as f:
        taxonomy_data = json.load(f)
    
    # Get all species from taxonomy
    all_species = get_all_species(taxonomy_data)
    
    # Get current labels from predictions
    current_labels = sorted(predictions_df['cropmodel_label'].unique())
    
    # Create a filter for labels
    selected_labels = st.multiselect(
        "Filter by current labels",
        options=current_labels,
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
    
    # Create a hierarchical selection for new labels
    st.write("Select a new label from the taxonomy:")
    
    # Add FalsePositive option
    label_options = ['FalsePositive'] + [f"{s['title']} ({s['scientificName']})" for s in all_species]
    new_label = st.selectbox(
        "Select new label for selected images",
        options=label_options
    )
    
    # Update button
    if st.button("Update Labels"):
        if not st.session_state.selected_images:
            st.warning("Please select at least one image")
        else:
            # Create new annotations for selected images
            new_annotations = []
            for img_id in st.session_state.selected_images:
                original_label = predictions_df.loc[predictions_df['crop_image_id'] == img_id, 'cropmodel_label'].iloc[0]
                new_annotations.append({
                    'image_id': img_id,
                    'original_label': original_label,
                    'new_label': new_label,
                    'timestamp': datetime.now().isoformat(),
                    'user': 'streamlit_user'  # Could be replaced with actual user authentication
                })
            
            # Append new annotations to the dataframe
            new_annotations_df = pd.DataFrame(new_annotations)
            annotations_df = pd.concat([annotations_df, new_annotations_df], ignore_index=True)
            
            # Save the updated annotations
            annotations_df.to_csv("app/data/annotations.csv", index=False)
            
            st.success(f"Added {len(st.session_state.selected_images)} new annotations")
            
            # Clear selections
            st.session_state.selected_images.clear()
            st.experimental_rerun()

if __name__ == "__main__":
    load_css()
    app() 
import streamlit as st
from pathlib import Path
from PIL import Image
from utils.styling import load_css
import pandas as pd
from utils.auth import require_login
from utils.annotations import load_annotations, apply_annotations, ensure_human_labeled
from utils.indices import load_predictions_indices, EFFECTIVE_PREDICTIONS_PATH
from utils.taxonomy import species_display, to_scientific


@st.cache_data
def _load_effective_predictions():
    if Path(EFFECTIVE_PREDICTIONS_PATH).exists():
        return pd.read_csv(EFFECTIVE_PREDICTIONS_PATH)
    return None


@st.cache_data
def _load_predictions_with_annotations():
    df = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")
    ann = load_annotations("app/data/annotations.csv")
    df = apply_annotations(df, ann, id_col="crop_image_id", label_col="cropmodel_label", set_col="set")
    return ensure_human_labeled(df, set_col="set")


@st.cache_data
def _load_indices():
    return load_predictions_indices()


def app():
    require_login()
    st.title("Predictions Viewer")

    # Load predictions (effective if available, else base + annotations)
    effective = _load_effective_predictions()
    if effective is not None:
        image_df = effective.copy()
    else:
        image_df = _load_predictions_with_annotations().copy()
    image_df["cropmodel_label"] = image_df["cropmodel_label"].astype(str)
    image_df = image_df[image_df["cropmodel_label"] != "FalsePositive"]

    # Detection score slider and human-labeled filter
    detection_threshold = st.slider("Detection Confidence Threshold",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.8,
                                    step=0.01,
                                    help="Filter by model detection confidence (0-1 scale)")
    human_labeled_only = st.checkbox(
        "Human-labeled only",
        value=True,
        help="Show only images that have been reviewed by a human")

    if image_df is None or image_df.empty:
        st.warning("No images found in experiments")
        return

    use_common = st.session_state.get("use_common_names", True)

    # Species list from index when available, else from (filtered) dataframe
    indices = _load_indices()
    base_filtered = image_df[image_df["human_labeled"] == True] if human_labeled_only else image_df
    if indices:
        species_list = indices["species_list"]
        # Default: first species in list that has data in base_filtered
        default_index = 0
        for i, sp in enumerate(species_list):
            if sp in base_filtered["cropmodel_label"].values:
                default_index = i
                break
    else:
        species_list = sorted(base_filtered["cropmodel_label"].unique().tolist())
        if not species_list:
            st.warning("No records match the current filters.")
            return
        default_species = base_filtered["cropmodel_label"].value_counts().index[0]
        default_index = species_list.index(default_species) if default_species in species_list else 0

    selected_species = st.selectbox(
        "Select a species",
        options=species_list,
        index=default_index,
        format_func=lambda x: species_display(x, use_common),
    )
    # Resolve to scientific name for filtering (dropdown shows common/scientific per toggle)
    selected_scientific = to_scientific(selected_species)

    # Filter by species (using index ids when available) then score and human_labeled
    if indices and selected_scientific in indices.get("by_species", {}):
        species_ids = set(indices["by_species"][selected_scientific])
        species_images = image_df[
            image_df["crop_image_id"].astype(str).isin(species_ids)
            & (image_df["score"] >= detection_threshold)
        ]
        if human_labeled_only:
            species_images = species_images[species_images["human_labeled"] == True]
    else:
        species_images = base_filtered[
            (base_filtered["cropmodel_label"] == selected_scientific)
            & (base_filtered["score"] >= detection_threshold)
        ]

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

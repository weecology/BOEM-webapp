import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import os
import json
import io
from datetime import datetime
from utils.styling import load_css
from utils.auth import require_login
from utils.annotations import load_annotations, reduce_annotations, apply_annotations, ensure_human_labeled
from utils.indices import load_predictions_indices, EFFECTIVE_PREDICTIONS_PATH
from utils.taxonomy import species_display, to_scientific

@st.cache_data
def get_all_species(taxonomy_data):
    """Extract all species from the taxonomy data."""
    species_list = []

    def extract_species(node):
        if node.get("rank") == "Species":
            species_list.append(
                {
                    "title": node.get("title"),
                    "scientificName": node.get("scientificName"),
                }
            )
        for child in node.get("children", []):
            extract_species(child)

    for node in taxonomy_data:
        extract_species(node)

    return species_list


@st.cache_data
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


@st.cache_data
def _load_effective_predictions():
    """Load effective predictions (annotations applied) when available."""
    if Path(EFFECTIVE_PREDICTIONS_PATH).exists():
        return pd.read_csv(EFFECTIVE_PREDICTIONS_PATH)
    return None


@st.cache_data
def _load_predictions_with_annotations():
    """Load base predictions and apply annotations (fallback when no effective CSV)."""
    df = pd.read_csv("app/data/most_recent_all_flight_predictions.csv")
    ann = load_annotations("app/data/annotations.csv")
    df = apply_annotations(df, ann, id_col="crop_image_id", label_col="cropmodel_label", set_col="set")
    return ensure_human_labeled(df, set_col="set")


@st.cache_data
def _load_indices():
    """Load pre-computed prediction indices for dropdowns and filtering."""
    return load_predictions_indices()


@st.cache_data
def _get_all_image_files():
    """Cache the full images directory listing; it is static while the app runs."""
    image_dir = Path("app/data/images")
    return list(image_dir.glob("*"))


@st.cache_data
def _load_taxonomy():
    """Load taxonomy data from JSON (cached)."""
    taxonomy_path = Path(__file__).resolve().parents[2] / "taxonomy.json"
    if not taxonomy_path.exists():
        taxonomy_path = Path(__file__).resolve().parents[1] / "data" / "taxonomy.json"
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def _get_all_species():
    """Get flattened list of all species from cached taxonomy."""
    taxonomy_data = _load_taxonomy()
    return get_all_species(taxonomy_data)


@st.cache_data
def _load_image_bytes(path_str: str):
    """Load full image file bytes once (no downscaling)."""
    img_path = Path(path_str)
    if not img_path.exists():
        return None
    try:
        return img_path.read_bytes()
    except OSError:
        return None


def _save_annotations(image_ids, new_label, mark_review=True):
    """Append annotations for given image_ids and save. Clears annotation cache."""
    predictions_df = st.session_state.get("bulk_labeling_predictions_df")
    if predictions_df is None or not image_ids:
        return 0
    annotations_df = load_or_create_annotations()
    predictions_df = predictions_df.copy()
    predictions_df["crop_image_id"] = predictions_df["crop_image_id"].astype(str)
    new_annotations = []
    for img_id in image_ids:
        row = predictions_df.loc[predictions_df["crop_image_id"] == img_id]
        if row.empty:
            continue
        original_label = row["cropmodel_label"].iloc[0]
        new_annotations.append({
            "image_id": img_id,
            "original_label": original_label,
            "new_label": new_label,
            "set": "review" if mark_review else None,
            "timestamp": datetime.now().isoformat(),
            "user": st.session_state.get("username", "streamlit_user"),
        })
    if not new_annotations:
        return 0
    new_df = pd.DataFrame(new_annotations)
    annotations_df = pd.concat([annotations_df, new_df], ignore_index=True)
    annotations_df.to_csv("app/data/annotations.csv", index=False)
    load_or_create_annotations.clear()
    return len(new_annotations)


def app():
    require_login()
    st.title("Bulk Image Labeling")
    st.text("Select multiple images and update their labels in bulk")

    # Confidence score slider
    confidence_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
        help="Filter by model detection confidence (0-1 scale)"
    )

    # Human-labeled filter
    human_labeled_only = st.checkbox(
        "Human-labeled only",
        value=False,
        help="Show only images that have been reviewed by a human"
    )

    # Load indices (for dropdowns and id-based filtering) and predictions
    indices = _load_indices()
    effective = _load_effective_predictions()
    if effective is not None:
        predictions_df = effective.copy()
    else:
        predictions_df = _load_predictions_with_annotations().copy()

    # Persist for fragment (avoids full rerun when only grid interacts)
    st.session_state["bulk_labeling_predictions_df"] = predictions_df

    # Dropdown options from index when available, else from dataframe
    use_common = st.session_state.get("use_common_names", True)
    if indices:
        current_labels = indices["species_list"]
        flights = indices.get("flight_list", [])
    else:
        current_labels = sorted(predictions_df["cropmodel_label"].dropna().unique().tolist())
        flights = sorted(predictions_df["flight_name"].dropna().unique().tolist()) if "flight_name" in predictions_df.columns else []

    # Load or create annotations dataframe (for saving / revert)
    annotations_df = load_or_create_annotations()
    latest_annotations = reduce_annotations(annotations_df)

    # Load taxonomy data (cached) and flatten to species list
    all_species = _get_all_species()

    # Create a filter for labels and flights
    selected_labels = st.multiselect(
        "Filter by current labels",
        options=current_labels,
        default=[],
        format_func=lambda x: species_display(x, use_common),
    )
    selected_flights = st.multiselect(
        "Filter by flight",
        options=flights,
        default=[]
    ) if flights else []

    # Resolve to scientific names for filtering (dropdown shows common/scientific per toggle)
    selected_labels_scientific = [to_scientific(l) for l in selected_labels]

    # Filter by labels/flights using index when available, else dataframe
    if indices and (selected_labels or selected_flights):
        filtered_ids = None
        if selected_labels:
            filtered_ids = set()
            for lab in selected_labels_scientific:
                filtered_ids.update(indices["by_species"].get(lab, []))
        else:
            filtered_ids = set(predictions_df["crop_image_id"].astype(str).tolist())
        if selected_flights:
            flight_ids = set()
            for fl in selected_flights:
                flight_ids.update(indices["by_flight"].get(fl, []))
            filtered_ids = (filtered_ids & flight_ids) if filtered_ids is not None else flight_ids
        cid = predictions_df["crop_image_id"].astype(str)
        filtered_df = predictions_df[cid.isin(filtered_ids)].copy()
    else:
        filtered_df = predictions_df.copy()
        if selected_labels:
            filtered_df = filtered_df[filtered_df["cropmodel_label"].isin(selected_labels_scientific)]
        if selected_flights:
            filtered_df = filtered_df[filtered_df["flight_name"].isin(selected_flights)]

    # Filter by confidence score and human-labeled
    if "score" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["score"] >= confidence_threshold]
    if human_labeled_only:
        filtered_df = filtered_df[filtered_df["human_labeled"] == True]
    
    # Get all images from the images directory (cached)
    image_files = _get_all_image_files()

    # Filter images that exist in our dataframe
    valid_images = [img for img in image_files if img.name in filtered_df['crop_image_id'].values]

    # Pagination controls
    st.subheader("Select Images to Relabel")
    total = len(valid_images)
    page_size = st.selectbox("Images per page", options=[24, 36, 48, 60], index=0)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    page_images = valid_images[start:end]

    st.caption(f"**{total}** image(s) for current filters — page **{page}** of **{total_pages}** (showing {len(page_images)} here)")

    # Session state for selection (used by fragment and main form)
    if "selected_images" not in st.session_state:
        st.session_state.selected_images = set()

    # Image grid in a fragment: only this block reruns on checkbox/button clicks
    @st.fragment
    def image_grid_fragment():
        # Handle quick FP save from fragment (e.g. "Mark FP" button)
        if st.session_state.get("pending_fp_save"):
            mark_review = st.session_state.get("bulk_mark_review", True)
            n = _save_annotations(list(st.session_state.selected_images), "FalsePositive", mark_review=mark_review)
            st.session_state.selected_images.clear()
            st.session_state.pending_fp_save = False
            st.toast(f"Marked {n} image(s) as FalsePositive")
            st.rerun()

        # Sync checkbox keys with selected_images so "Select all" / "Clear all" show correctly
        for img_path in page_images:
            st.session_state[f"select_{img_path.name}"] = img_path.name in st.session_state.selected_images

        # Quick actions
        if st.button("Select all in current page", key="select_all_page"):
            for img_path in page_images:
                st.session_state.selected_images.add(img_path.name)
                st.session_state[f"select_{img_path.name}"] = True

        if st.button("Clear all selections", key="clear_all_selections"):
            st.session_state.selected_images.clear()
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith("select_"):
                    del st.session_state[k]

        cols = st.columns(4)
        for idx, img_path in enumerate(page_images):
            with cols[idx % 4]:
                try:
                    img_bytes = _load_image_bytes(str(img_path))
                    if img_bytes is not None:
                        st.image(img_bytes, use_container_width=True)
                    else:
                        st.warning(f"Image not found: {img_path.name}")

                    # Checkbox: state synced above from selected_images
                    if st.checkbox("Select", key=f"select_{img_path.name}"):
                        st.session_state.selected_images.add(img_path.name)
                    else:
                        st.session_state.selected_images.discard(img_path.name)

                    # One-click FalsePositive for rapid relabeling
                    if st.button("Mark FP", key=f"fp_{img_path.name}", type="secondary"):
                        st.session_state.selected_images.add(img_path.name)
                        st.session_state.pending_fp_save = True
                except Exception as e:
                    st.error(f"Error loading image {img_path.name}: {str(e)}")

    image_grid_fragment()
    
    # New label selection
    st.subheader("Update Labels")
    
    # Quick action: save selected as FalsePositive (no dropdown)
    if st.session_state.selected_images:
        st.caption(f"{len(st.session_state.selected_images)} image(s) selected")
    if st.button("Save selected as FalsePositive", type="primary", key="save_fp_bulk"):
        if st.session_state.selected_images:
            mark_review = st.session_state.get("bulk_mark_review", True)
            n = _save_annotations(list(st.session_state.selected_images), "FalsePositive", mark_review=mark_review)
            st.success(f"Marked {n} image(s) as FalsePositive")
            st.session_state.selected_images.clear()
            st.rerun()
        else:
            st.warning("Select at least one image first")

    st.write("Or choose another label:")
    label_options = ['FalsePositive'] + [f"{s['title']} ({s['scientificName']})" for s in all_species]
    new_label = st.selectbox(
        "Select new label for selected images",
        options=label_options
    )
    
    # Option to mark QC as review (persisted for fragment's quick FP save)
    mark_review = st.checkbox(
        "Mark QC as review",
        value=st.session_state.get("bulk_mark_review", True),
        help="When enabled, saved rows will set set='review'.",
        key="bulk_mark_review_cb",
    )
    st.session_state["bulk_mark_review"] = mark_review

    if st.button("Update Labels"):
        if not st.session_state.selected_images:
            st.warning("Please select at least one image")
        else:
            n = _save_annotations(list(st.session_state.selected_images), new_label, mark_review=mark_review)
            st.success(f"Added {n} new annotation(s)")
            st.session_state.selected_images.clear()
            st.rerun()

    # Recent history and revert (optional)
    with st.expander("Recent edits"):
        if latest_annotations.empty:
            st.info("No annotations yet.")
        else:
            # Show last 50 edits; use "Species" column names and display names for labels
            show_df = latest_annotations.sort_values("timestamp", ascending=False).head(50)
            display_df = show_df.copy()
            display_df = display_df.rename(columns={"original_label": "Original Species", "new_label": "New Species"})
            display_df["Original Species"] = display_df["Original Species"].map(lambda s: species_display(s, use_common) if pd.notna(s) else s)
            display_df["New Species"] = display_df["New Species"].map(lambda s: species_display(s, use_common) if pd.notna(s) else s)
            st.dataframe(display_df, use_container_width=True)

            # Simple revert by image_id (use show_df with original column names)
            revert_image = st.selectbox("Revert image", options=[""] + show_df["image_id"].astype(str).tolist())
            if revert_image:
                row = show_df.loc[show_df["image_id"].astype(str) == revert_image].iloc[0]
                # Determine original label to revert to
                revert_to = row.get("original_label") if pd.notna(row.get("original_label")) else None
                if not revert_to:
                    # Fallback to current predictions label
                    if revert_image in predictions_df['crop_image_id'].astype(str).values:
                        revert_to = predictions_df.loc[predictions_df['crop_image_id'].astype(str) == revert_image, 'cropmodel_label'].iloc[0]
                if revert_to:
                    revert_row = {
                        'image_id': revert_image,
                        'original_label': predictions_df.loc[predictions_df['crop_image_id'].astype(str) == revert_image, 'cropmodel_label'].iloc[0] if revert_image in predictions_df['crop_image_id'].astype(str).values else row.get('new_label'),
                        'new_label': revert_to,
                        'set': 'review' if mark_review else None,
                        'timestamp': datetime.now().isoformat(),
                        'user': st.session_state.get('username', 'streamlit_user')
                    }
                    annotations_df = pd.concat([annotations_df, pd.DataFrame([revert_row])], ignore_index=True)
                    annotations_df.to_csv("app/data/annotations.csv", index=False)
                    # Refresh cached annotations on next rerun
                    load_or_create_annotations.clear()
                    st.success(f"Reverted {revert_image} to {revert_to}")
                    st.rerun()

if __name__ == "__main__":
    load_css()
    app() 
import streamlit as st
from pathlib import Path
from utils.styling import load_css
from utils.auth import require_login
from utils.indices import load_predictions_indices


VIDEOS_DIR = Path("app/data/videos")


@st.cache_data
def _flight_list():
    """Flight list from predictions index or fallback to empty (user should run prepare.py)."""
    indices = load_predictions_indices()
    if indices and indices.get("flight_list"):
        return indices["flight_list"]
    return []


def app():
    require_login()
    st.title("Detection flythrough videos")
    st.markdown("Select a flight to play its detection flythrough video (detections overlaid on imagery).")

    with st.expander("How these videos are made"):
        st.markdown("""
        The **SEABIRD camera system** has seven cameras. For each flight we select the camera with the most detections,
        thin empty frames by a factor of 4, and remove any stretches longer than 30 seconds without detections.
        The result is a shorter flythrough video focused on frames where the model detected wildlife.
        """)

    flights = _flight_list()
    if not flights:
        st.warning("No flights in the predictions index. Run **prepare.py** to load data and download videos.")
        return

    selected = st.selectbox(
        "Flight",
        options=flights,
        format_func=lambda x: x,
        key="flight_video_select",
        help="Choose a flight to view its detection flythrough video.",
    )
    if not selected:
        return

    video_filename = f"{selected}_flythrough.avi"
    video_path = VIDEOS_DIR / video_filename
    if not video_path.exists():
        st.info(
            f"Video for **{selected}** is not available yet. Run **prepare.py** to download "
            "flythrough videos from Comet (only the latest video per flight is stored)."
        )
        return

    st.video(str(video_path))
    st.caption(f"**{selected}** — detection flythrough (`.avi`).")


if __name__ == "__main__":
    load_css()
    app()

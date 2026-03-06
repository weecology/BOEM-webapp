import streamlit as st
from pathlib import Path
from utils.styling import load_css
from utils.auth import require_login
from utils.indices import load_predictions_indices


# Resolve relative to this file so it works regardless of Streamlit cwd
VIDEOS_DIR = Path(__file__).resolve().parents[1] / "data" / "videos"


@st.cache_data
def _flight_list(_indices_mtime: float = 0):
    """Flight list from predictions index (Gulf-only when built by prepare.py).
    _indices_mtime is used only to invalidate cache when app/data/predictions_indices.json is updated (e.g. after deploy).
    """
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

    # Invalidate cache when predictions_indices.json is updated (e.g. after deploy or re-running prepare.py)
    indices_path = Path(__file__).resolve().parents[1] / "data" / "predictions_indices.json"
    indices_mtime = indices_path.stat().st_mtime if indices_path.exists() else 0.0
    flights = _flight_list(_indices_mtime=indices_mtime)
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

    video_filename = f"{selected}_flythrough.mp4"
    video_path = VIDEOS_DIR / video_filename
    if not video_path.exists():
        st.info(
            f"Video for **{selected}** is not available yet. Run **prepare.py** to download "
            "flythrough videos from Comet (only the latest video per flight is stored)."
        )
        return

    st.video(str(video_path))
    st.caption(f"**{selected}** — detection flythrough (`.mp4`).")

    with open(video_path, "rb") as f:
        video_bytes = f.read()
    st.download_button(
        label="Download video",
        data=video_bytes,
        file_name=video_filename,
        mime="video/mp4",
        key="flight_video_download",
    )


if __name__ == "__main__":
    load_css()
    app()

import streamlit as st
from pathlib import Path
import cv2
from PIL import Image
import numpy as np

def app():
    st.title("Flight Line Video Viewer")

    # Create video directory path 
    video_dir = Path("app/data/videos")

    # Get list of video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        video_files.extend(list(video_dir.glob(f"*{ext}")))

    if not video_files:
        st.warning("No video files found in the videos directory")
        return

    # Create dropdown to select video
    selected_video = st.selectbox(
        "Select a flight line video to view",
        [v.name for v in video_files],
        format_func=lambda x: x.split('.')[0]  # Remove file extension in display
    )

    if selected_video:
        video_path = video_dir / selected_video
        
        # Create a video player
        video_file = open(str(video_path), 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

if __name__ == "__main__":
    app()

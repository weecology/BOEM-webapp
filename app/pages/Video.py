import streamlit as st
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from utils.styling import load_css

def app():
    st.title("Flight Line Video Viewer")

    # Create video directory path 
    video_dir = Path("app/data/")

    # Get list of video files
    video_files = []
    for ext in ['.mp4']:
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
        
        try:
            # Verify video can be opened
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                st.error("Error: Could not open video file")
                return
            cap.release()
            
            # Create a video player with explicit mime type
            video_file = open(str(video_path), 'rb')
            video_bytes = video_file.read()
            video_file.close()
            
            # Try to play video with mime type specification
            st.video(video_bytes, format='video/mp4')
            
            # Add a download button as fallback
            st.download_button(
                label="Download video",
                data=video_bytes,
                file_name=selected_video,
                mime="video/mp4"
            )
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    load_css()
    app()

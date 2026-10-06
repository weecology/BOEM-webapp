"""Shared sidebar navigation.

The app hides Streamlit's automatic multipage nav (see .streamlit/config.toml
-> [ui] hide_sidebar_nav = true) and previously replaced it with nothing, so
every page other than the landing page was reachable only by typing its URL
directly. This renders a real nav in its place, grouped to match how the
project is meant to read: a public "Explore" section up front, and an
"Internal Tools" section for model-development/monitoring pages.

Call render_sidebar_nav() near the top of every page (main.py and each file
in app/pages/), before or after require_login() — it works either way since
it only renders links, it doesn't gate anything itself.
"""
import streamlit as st

# (path relative to app/main.py, label, icon)
EXPLORE_PAGES = [
    ("main.py", "Overview", "🌊"),
    ("pages/Analysis.py", "Species Analysis", "📊"),
    ("pages/Image_viewer.py", "Predictions Viewer", "🖼️"),
    ("pages/Flight_Reports.py", "Flight Reports", "📄"),
    ("pages/Flight_Videos.py", "Flight Videos", "🎥"),
]

INTERNAL_PAGES = [
    ("pages/Model_Development.py", "Model Development", "🧪"),
    ("pages/Bulk_Labeling.py", "Bulk Labeling", "🏷️"),
]


def render_sidebar_nav():
    """Render the grouped page-link nav in the sidebar."""
    with st.sidebar:
        st.caption("Explore")
        for path, label, icon in EXPLORE_PAGES:
            st.page_link(path, label=label, icon=icon)

        st.divider()
        st.caption("Internal tools")
        for path, label, icon in INTERNAL_PAGES:
            st.page_link(path, label=label, icon=icon)

        st.divider()

import streamlit as st
from pathlib import Path
import zipfile
import io
from utils.styling import load_css
from utils.auth import require_login
from utils.indices import load_predictions_indices


REPORTS_DIR = Path(__file__).resolve().parents[1] / "data" / "reports"


@st.cache_data
def _flight_list(_indices_mtime: float = 0):
    """Flight list from predictions index (Gulf-only when built by prepare.py)."""
    indices = load_predictions_indices()
    if indices and indices.get("flight_list"):
        return indices["flight_list"]
    return []


def _zip_report_folder(folder: Path) -> bytes:
    """Build a zip of all files under folder (relative paths preserved)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in folder.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(folder))
    buf.seek(0)
    return buf.getvalue()


def app():
    require_login()
    st.title("Flight reports")
    st.markdown("Select a flight to view its transect map and download the full report assets (HTML map, PDF, shapefiles, CSV).")

    indices_path = Path(__file__).resolve().parents[1] / "data" / "predictions_indices.json"
    indices_mtime = indices_path.stat().st_mtime if indices_path.exists() else 0.0
    flights = _flight_list(_indices_mtime=indices_mtime)
    if not flights:
        st.warning("No flights in the predictions index. Run **prepare.py** to load data and download reports.")
        return

    selected = st.selectbox(
        "Flight",
        options=flights,
        format_func=lambda x: x,
        key="flight_report_select",
        help="Choose a flight to view its report and transect map.",
    )
    if not selected:
        return

    report_dir = REPORTS_DIR / selected
    if not report_dir.is_dir():
        st.info(
            f"Report for **{selected}** is not available yet. Run **prepare.py** to download "
            "flight reports from Comet (only the latest report per flight is stored)."
        )
        return

    # Find transect_map.html (may be in report_dir or one subdir)
    html_candidates = list(report_dir.glob("**/transect_map.html"))
    html_path = html_candidates[0] if html_candidates else (report_dir / "transect_map.html")
    if not html_path.exists():
        st.warning(f"No **transect_map.html** found for **{selected}**. Report folder has: {[p.name for p in report_dir.iterdir()]}.")
    else:
        st.subheader("Transect map")
        try:
            html_content = html_path.read_text(encoding="utf-8", errors="replace")
            st.components.v1.html(html_content, height=700, scrolling=True)
        except Exception as e:
            st.error(f"Could not render map: {e}")
            st.caption("Download the report folder below and open transect_map.html in a browser for full fidelity.")

    # Download entire report folder as zip
    st.subheader("Download report assets")
    zip_bytes = _zip_report_folder(report_dir)
    zip_name = f"{selected}_report.zip"
    st.download_button(
        label="Download report folder (ZIP)",
        data=zip_bytes,
        file_name=zip_name,
        mime="application/zip",
        key="flight_report_download",
        help="Download all report assets for this flight (transect map HTML, report PDF, shapefiles, observations CSV).",
    )


if __name__ == "__main__":
    load_css()
    app()

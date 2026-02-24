import streamlit as st

def load_css():
    # Load USWDS base CSS (used on Analysis, Image viewer, Bulk Labeling, Model Development)
    st.markdown('<link rel="stylesheet" href="static/css/uswds.min.css">', unsafe_allow_html=True)
    st.markdown(f'<style>{_scientific_theme_css()}</style>', unsafe_allow_html=True)


def inject_landing_css():
    """Use on landing page only: scientific theme without USWDS for a cleaner look."""
    st.markdown(f'<style>{_scientific_theme_css()}</style>', unsafe_allow_html=True)


def _scientific_theme_css():
    return """
        /* ----- Scientific theme: typography & spacing ----- */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .main h1 {
            font-family: 'Georgia', 'Cambria', 'Times New Roman', serif;
            font-weight: 600;
            color: #1b3a57;
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
        }
        .main h2 {
            font-family: 'Georgia', 'Cambria', 'Times New Roman', serif;
            font-weight: 600;
            color: #1b3a57;
            font-size: 1.35rem;
            margin-top: 1.25rem;
            margin-bottom: 0.5rem;
            padding-bottom: 0.25rem;
            border-bottom: 1px solid #e5e7eb;
        }
        .main h3 {
            font-size: 1.1rem;
            font-weight: 600;
            color: #374151;
            margin-top: 0.75rem;
            margin-bottom: 0.35rem;
        }
        .main p, .main .stMarkdown {
            line-height: 1.5;
            margin-bottom: 0.5rem;
        }
        .main .element-container {
            margin-bottom: 0.5rem;
        }

        /* ----- Figure highlight: make images/figures stand out ----- */
        .main [data-testid="stImage"] {
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 4px 14px rgba(27, 58, 87, 0.12);
            border: 1px solid #c7d2e0;
            background: #fff;
        }
        .main [data-testid="stImage"] img {
            border-radius: 6px;
        }
        .main figcaption, .main .stCaption {
            font-size: 0.9rem;
            color: #4b5563;
            margin-top: 0.35rem;
            font-style: italic;
        }

        /* ----- Compact metrics / stats ----- */
        .main [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 0.6rem 0.75rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }

        /* ----- Select styling (align with theme) ----- */
        .stSelectbox > div > div {
            font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.95rem;
            padding: 0.5rem;
            border: 1px solid #94a3b8;
            border-radius: 4px;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 16L4 8h16z' fill='%231b3a57'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 0.6rem center;
            background-size: 0.65rem;
        }
        .stSelectbox > div > div:hover {
            border-color: #1b3a57;
        }
        .stSelectbox > div > div:focus {
            outline: 2px solid #1b3a57;
            outline-offset: 0;
        }
        .stSelectbox label {
            font-size: 0.95rem;
            color: #374151;
        }
    """ 
import streamlit as st

def load_css():
    # Load USWDS base CSS
    st.markdown('<link rel="stylesheet" href="static/css/uswds.min.css">', unsafe_allow_html=True)
    
    # Add our custom styles
    custom_css = """
        /* Select styling */
        .stSelectbox > div > div {
            font-family: Source Sans Pro Web, Helvetica Neue, Helvetica, Roboto, Arial, sans-serif;
            font-size: 1.06rem;
            line-height: 1;
            padding: 0.5rem;
            border: 2px solid #565c65;
            border-radius: 0.25rem;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 16L4 8h16z' fill='%23565c65'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 0.75rem center;
            background-size: 0.75rem;
            white-space: nowrap;
            overflow: visible;
            text-overflow: clip;
            height: auto;
            min-height: 2.5rem;
        }
        
        .stSelectbox > div > div:hover {
            border-color: #005ea2;
        }
        
        .stSelectbox > div > div:focus {
            outline: 0.25rem solid #2491ff;
            outline-offset: 0;
        }
        
        .stSelectbox label {
            font-family: Source Sans Pro Web, Helvetica Neue, Helvetica, Roboto, Arial, sans-serif;
            font-size: 1.06rem;
            line-height: 1.1;
            margin-bottom: 0.5rem;
        }
    """
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True) 
import os
import streamlit as st


def get_configured_credentials():
    """Return configured username/password from Streamlit secrets or env vars.

    Precedence:
    1. st.secrets["auth"]["username"], st.secrets["auth"]["password"]
    2. ENV: APP_USERNAME, APP_PASSWORD
    3. Default None if not configured
    """
    username = None
    password = None

    try:
        auth_secrets = st.secrets.get("auth", {})
        username = auth_secrets.get("username")
        password = auth_secrets.get("password")
    except Exception:
        # st.secrets might not be configured locally
        pass

    if not username:
        username = os.getenv("APP_USERNAME")
    if not password:
        password = os.getenv("APP_PASSWORD")

    if username and password:
        return username, password
    return None, None


def is_authenticated() -> bool:
    """Check if user is authenticated in session."""
    return bool(st.session_state.get("authenticated", False))


def logout():
    """Clear authentication from session."""
    for key in ["authenticated", "username"]:
        if key in st.session_state:
            del st.session_state[key]


def login_form():
    """Render a simple login form. Returns True when login succeeded."""
    configured_username, configured_password = get_configured_credentials()

    if not configured_username or not configured_password:
        st.warning("Admin has not configured login credentials. Set `.streamlit/secrets.toml` or APP_USERNAME/APP_PASSWORD.")

    st.markdown("### Login")
    username_input = st.text_input("Username", key="login_username")
    password_input = st.text_input("Password", type="password", key="login_password")

    submit = st.button("Sign in")

    if submit:
        if configured_username and configured_password and username_input == configured_username and password_input == configured_password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username_input
            st.success("Logged in successfully.")
            st.rerun()
        else:
            st.error("Invalid credentials.")
    return is_authenticated()


def require_login():
    """Call at the top of pages. If not authenticated, show login and stop."""
    if is_authenticated():
        with st.sidebar:
            st.caption(f"Signed in as: {st.session_state.get('username', '')}")
            if st.button("Log out"):
                logout()
                st.rerun()
        return

    st.sidebar.info("Please sign in to continue.")
    logged_in = login_form()
    if not logged_in:
        st.stop()



"""
Streamlit Herbalism Research Chat Application

This application integrates Gemini models, file upload capabilities,
and Google Custom Search API for herbalism research.
"""
import os

import streamlit as st
# Load components
from components.chat import render_chat_settings_tab, render_chat_tab
from components.document_upload import render_document_upload_tab
from components.search import render_search_tab
from components.visualization import render_visualization_tab
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for required API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

# Set page configuration
st.set_page_config(
    page_title="Herbalism Research Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def check_api_keys():
    """Check if required API keys are set and display warnings if not."""
    missing_keys = []

    if not GEMINI_API_KEY:
        missing_keys.append("GEMINI_API_KEY")

    if not GOOGLE_SEARCH_API_KEY:
        missing_keys.append("GOOGLE_SEARCH_API_KEY")

    if not GOOGLE_SEARCH_ENGINE_ID:
        missing_keys.append("GOOGLE_SEARCH_ENGINE_ID")

    if missing_keys:
        st.warning(
            f"Missing API keys: {', '.join(missing_keys)}. "
            f"Please add them to your .env file."
        )

        # Show instructions
        with st.expander("How to set up API keys"):
            st.markdown(
                """
            ### Setting up API Keys
            
            1. Create a `.env` file in the root directory of this project
            2. Add the following lines to the file:
            ```
            GEMINI_API_KEY=your_gemini_api_key_here
            GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here
            GOOGLE_SEARCH_ENGINE_ID=your_google_search_engine_id_here
            ```
            
            #### Getting API Keys
            
            - **Gemini API Key**: Get it from [Google AI Studio](https://makersuite.google.com/app/apikey)
            - **Google Search API Key**: Get it from [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
            - **Google Search Engine ID**: Create a programmable search engine at [Programmable Search Engine](https://programmablesearchengine.google.com/about/)
            """
            )

        return False

    return True


def custom_css():
    """Add custom CSS to the application."""
    st.markdown(
        """
    <style>
    .stApp {
        max-width: -webkit-fill-available;
        margin: 0 auto;
    }
    
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Chat styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
    }
    
    .assistant-message {
        background-color: #f6ffed;
        border-left: 5px solid #52c41a;
    }
    
    /* Card styling */
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .custom-button {
        background-color: #1890ff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f9f9f9;
    }
    
    /* Right sidebar styling */
    [data-testid="stSidebar"] {
        background-color: darkslategray;
    }
    
    /* Make document section in right sidebar more compact */
    [data-testid="stVerticalBlock"] > div:nth-child(2) h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    /* Improve chat layout */
    .stChatInputContainer {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding-top: 1rem;
        border-top: 1px solid #f0f0f0;
        z-index: 100;
    }
    
    .stChatMessageContent {
        padding: 0.75rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/herbal-medicine.png", width=80)
        st.title("Herbalism Research Assistant")

        st.markdown("---")

        # About section
        st.subheader("About")
        st.markdown(
            "This application helps with herbalism research by providing "
            "chat capabilities, document analysis, search functionality, "
            "and data visualization."
        )

        st.markdown("---")

        # API key status
        st.subheader("API Status")

        if GEMINI_API_KEY:
            st.success("Gemini API: Connected")
        else:
            st.error("Gemini API: Not Connected")

        if GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID:
            st.success("Google Search API: Connected")
        else:
            st.error("Google Search API: Not Connected")

        st.markdown("---")

        # Credits
        st.caption("Developed with Streamlit and Google Gemini")
        st.caption("© 2025 Herbalism Research Assistant")


def main():
    """Main application function."""
    # Add custom CSS
    custom_css()

    # Render sidebar
    sidebar()

    # Check API keys
    api_keys_valid = check_api_keys()

    # Create a layout with main content and right sidebar
    main_col, right_sidebar = st.columns([3, 1])

    with main_col:
        # Create tabs for the main content area
        tabs = st.tabs(["Chat", "Search", "Visualizations", "Settings"])

        # Render tabs
        with tabs[0]:
            if api_keys_valid or not GEMINI_API_KEY:
                render_chat_tab()
            else:
                st.error("Chat functionality requires API keys to be set.")

        with tabs[1]:
            if api_keys_valid or not (
                GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID
            ):
                render_search_tab()
            else:
                st.error("Search functionality requires API keys to be set.")

        with tabs[2]:
            render_visualization_tab()

        with tabs[3]:
            if api_keys_valid or not GEMINI_API_KEY:
                render_chat_settings_tab()
            else:
                st.error("Settings functionality requires API keys to be set.")

    # Right sidebar for documents
    with right_sidebar:
        st.markdown("### Documents")
        render_document_upload_tab()


if __name__ == "__main__":
    main()

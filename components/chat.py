"""
Chat interface component for the Streamlit application.
"""
import json
from typing import Any, Dict, List, Optional

import streamlit as st
from services.gemini_service import gemini_service
from utils.context_management import ContextManager


def initialize_chat_state():
    """Initialize the chat state in the session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "context_manager" not in st.session_state:
        # Load the system prompt from the session state if available
        system_prompt = None
        if "system_prompt" in st.session_state:
            system_prompt = st.session_state.system_prompt

        st.session_state.context_manager = ContextManager(system_prompt)


def display_chat_message(role: str, content: str, is_user: bool):
    """
    Display a chat message in the Streamlit interface.

    Args:
        role: Role of the message sender
        content: Message content
        is_user: Whether the message is from the user
    """
    with st.chat_message(role):
        st.markdown(content)


def display_chat_history():
    """Display the chat history in the Streamlit interface."""
    for message in st.session_state.messages:
        display_chat_message(
            message["role"], message["content"], message["role"] == "user"
        )


def add_user_message(content: str):
    """
    Add a user message to the chat history.

    Args:
        content: Message content
    """
    # Add to session state
    st.session_state.messages.append({"role": "user", "content": content})

    # Add to context manager
    st.session_state.context_manager.conversation.add_user_message(content)


def add_assistant_message(content: str):
    """
    Add an assistant message to the chat history.

    Args:
        content: Message content
    """
    # Add to session state
    st.session_state.messages.append({"role": "assistant", "content": content})

    # Add to context manager
    st.session_state.context_manager.conversation.add_assistant_message(content)


def generate_response(user_input: str) -> str:
    """
    Generate a response from the Gemini model.

    Args:
        user_input: User input text

    Returns:
        Generated response
    """
    # Prepare context for the model
    messages = st.session_state.context_manager.prepare_context_for_model()

    # Generate response
    response = gemini_service.generate_response(messages)

    return response


def chat_interface():
    """Render the chat interface component."""
    st.title("Herbalism Research Chat")

    # Initialize chat state
    initialize_chat_state()

    # Display chat history
    display_chat_history()

    # Chat input
    if user_input := st.chat_input("Ask about herbalism..."):
        # Add user message to chat
        add_user_message(user_input)
        display_chat_message("user", user_input, True)

        with st.spinner("Thinking..."):
            # Generate response
            response = generate_response(user_input)

            # Add assistant message to chat
            add_assistant_message(response)
            display_chat_message("assistant", response, False)


def system_prompt_editor():
    """Render the system prompt editor component."""
    st.subheader("System Prompt")

    # Initialize system prompt if not in session state
    if "system_prompt" not in st.session_state:
        # Default system prompt
        default_prompt = {
            "role": "research assistant",
            "name": "teaGPT",
            "specialty": "clinical herbalism"
            # Add more fields as needed
        }
        st.session_state.system_prompt = default_prompt

    # Display current system prompt
    system_prompt_json = json.dumps(st.session_state.system_prompt, indent=2)

    # Editor for system prompt
    updated_prompt = st.text_area(
        "Edit System Prompt (JSON)", value=system_prompt_json, height=300
    )

    # Update button
    if st.button("Update System Prompt"):
        try:
            # Parse the JSON
            new_prompt = json.loads(updated_prompt)

            # Update session state
            st.session_state.system_prompt = new_prompt

            # Update context manager
            if "context_manager" in st.session_state:
                st.session_state.context_manager = ContextManager(new_prompt)

            # Update Gemini service
            gemini_service.set_system_prompt(new_prompt)

            st.success("System prompt updated successfully!")

        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")


def chat_settings():
    """Render the chat settings component."""
    st.subheader("Chat Settings")

    # Model selection
    available_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
    selected_model = st.selectbox("Model", options=available_models, index=0)

    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic",
    )

    # Max tokens slider
    max_tokens = st.slider(
        "Max Output Tokens",
        min_value=256,
        max_value=4096,
        value=2048,
        step=256,
        help="Maximum number of tokens in the response",
    )

    # Update settings button
    if st.button("Apply Settings"):
        # Update Gemini service
        if gemini_service.api_available and gemini_service.model is not None:
            try:
                gemini_service.model_name = selected_model
                # Create a new model instance with the selected model name
                gemini_service.model = genai.GenerativeModel(selected_model)
                st.success("Chat settings updated successfully!")
            except Exception as e:
                st.error(f"Error updating model: {e}")
        else:
            st.warning("Gemini API is not available. Settings cannot be applied.")

    # Clear chat history button
    if st.button("Clear Chat History"):
        # Clear session state
        st.session_state.messages = []

        # Clear context manager
        if "context_manager" in st.session_state:
            st.session_state.context_manager.conversation.clear()

        st.success("Chat history cleared!")


def render_chat_tab():
    """Render the chat tab in the Streamlit application."""
    chat_interface()


def render_chat_settings_tab():
    """Render the chat settings tab in the Streamlit application."""
    system_prompt_editor()
    st.divider()
    chat_settings()

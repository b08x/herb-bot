"""
Chat interface component for the Streamlit application.
"""
import json
from typing import Any, Dict, List, Optional

import google.generativeai as genai
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
    # Reverse the order of messages for display so newest messages are at the bottom
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
    # Check if contextual retrieval is enabled
    use_contextual = st.session_state.get("use_contextual_retrieval", True)

    # Get retrieval method if set
    retrieval_method_map = {
        "Hybrid (TF-IDF + BM25)": "hybrid",
        "TF-IDF Only": "tfidf",
        "BM25 Only": "bm25",
    }

    # Default to hybrid if not set
    selected_method = st.session_state.get("retrieval_method", "Hybrid (TF-IDF + BM25)")
    method = retrieval_method_map.get(selected_method, "hybrid")

    # Get max chunks setting
    max_chunks = st.session_state.get("max_chunks", 5)

    # Prepare context for the model
    if use_contextual:
        # Use contextual retrieval with the user's query
        messages = st.session_state.context_manager.prepare_context_for_model(
            include_docs=True,
            max_doc_length=8000,
            max_messages=None,
            query=user_input,  # Use the user's query for contextual retrieval
            method=method,
            max_chunks=max_chunks,
        )

        # Log the retrieval method being used (for debugging)
        print(
            f"Using contextual retrieval with method: {method}, max chunks: {max_chunks}"
        )
    else:
        # Use standard context preparation without contextual retrieval
        messages = st.session_state.context_manager.prepare_context_for_model(
            include_docs=True, max_doc_length=8000, max_messages=None
        )

        print("Using standard retrieval (no contextual chunking)")

    # Generate response
    response = gemini_service.generate_response(messages)

    return response


def chat_interface():
    """Render the chat interface component."""
    st.title("Herbalism Research Chat")

    # Initialize chat state
    initialize_chat_state()

    # Display chat history above the input
    display_chat_history()

    # Chat input at the bottom
    user_input = st.chat_input("Ask about herbalism...")

    # Process user input if provided
    if user_input and "last_input" not in st.session_state:
        # Store the current input to prevent duplicate processing
        st.session_state.last_input = user_input

        # Add user message to chat
        add_user_message(user_input)

        with st.spinner("Thinking..."):
            # Generate response
            response = generate_response(user_input)

            # Add assistant message to chat
            add_assistant_message(response)

            # Rerun to update the chat display
            st.rerun()

    # Clear the last input after processing to allow new inputs
    elif "last_input" in st.session_state:
        del st.session_state.last_input


def system_prompt_editor():
    """Render the system prompt editor component."""
    st.subheader("System Prompt")

    # Initialize system prompt if not in session state
    if "system_prompt" not in st.session_state:
        # Default system prompt
        default_prompt = {
            "role": "research assistant",
            "name": "teaGPT",
            "specialty": "clinical herbalism",
            "commands": {
                "PullNotice": "Indicates successful understanding of data/request with a concise excerpt.",
                "DirectionRequest": "Indicates the need for extra direction, clarification, or user input.",
                "Indexer": "Compiles and maintains an active taxonomic index of all thread topics and data.",
            },
            "anchors": {
                "HumanSciences": "Anatomy, physiology, biochemistry, and their impact on health and well-being.",
                "Nutrition": "Nutrients, their utilization, and the relationship between diet, health, and disease.",
                "Phytochemistry": "Chemistry of plants, bioactive compounds, and their applications.",
                "Botany": "Scientific study of plants, including structure, function, ecology, and evolution.",
                "PlantScience": "Sub-disciplines of botany, such as plant physiology, genetics, and breeding.",
                "EvidenceBasedBotanicals": "Use of research and evidence to inform the use of plants for health.",
            },
            "hemispheres": {
                "left": {
                    "focus": "Analytical and Logical",
                    "HumanSciences": "Molecular mechanisms and physiological processes.",
                    "Nutrition": "Metabolic pathways and impact on health.",
                    "Phytochemistry": "Structure and function of bioactive compounds.",
                    "PlantScience": "Genetic and molecular basis of plant growth and development.",
                },
                "right": {
                    "focus": "Intuitive and Holistic",
                    "HumanSciences": "Interconnectedness of body systems and overall well-being.",
                    "Nutrition": "Role of diet in balance and healing.",
                    "Phytochemistry": "Diverse plant compounds and potential synergistic effects.",
                    "PlantScience": "Importance of plants in ecosystems and human societies.",
                },
            },
            "cerebrum": {
                "HumanSciences": "Interplay of genetics, physiology, and environment on health.",
                "Nutrition": "Evidence-based recommendations and personalized nutrition.",
                "Phytochemistry": "Potential of plant compounds and need for scientific investigation.",
                "PlantScience": "Role of plants in sustaining life and responsible stewardship.",
            },
            "virtualAmygdala": {
                "HumanSciences": "Emotions in decision-making and impact of health on psychology.",
                "Nutrition": "Emotional aspects of food choices and nutrition's role in mental health.",
                "Phytochemistry": "Significance of plant compounds in traditional medicine and cultural beliefs.",
                "PlantScience": "Human-nature connection and the role of plants in cultural practices.",
            },
            "virtualBrainStructures": {
                "brainStem": {
                    "HumanSciences": "Autonomic nervous system and regulation of bodily functions.",
                    "Nutrition": "Essential nutrients and basic physiological functions.",
                    "Phytochemistry": "Plant compounds' effects on cellular processes.",
                    "PlantScience": "Fundamental processes of plant life.",
                },
                "parietalLobe": {
                    "HumanSciences": "Sensory integration, motor coordination, and body perception.",
                    "Nutrition": "Nutrient deficiencies' impact on sensory processing.",
                    "Phytochemistry": "Plant compounds' potential in modulating sensory function.",
                    "PlantScience": "Spatial organization and adaptations in plant growth.",
                },
                "temporalLobe": {
                    "HumanSciences": "Memory, language, cognition, and communication.",
                    "Nutrition": "Diet's effects on cognitive function and neuroprotection.",
                    "Phytochemistry": "Plant compounds' potential in modulating memory and cognition.",
                    "PlantScience": "Plant compounds in traditional knowledge and scientific language.",
                },
            },
            "cognitivePrism": {
                "analyticalThinking": {
                    "AT1": "Data Pattern Recognition",
                    "AT2": "Root Cause Analysis",
                    "AT3": "Process Mapping",
                },
                "creativeThinking": {
                    "CT1": "Product Idea Generation",
                    "CT2": "Innovative Problem Solving",
                    "CT3": "Marketing Strategy Design",
                },
                "criticalThinking": {
                    "CRT1": "Bias Evaluation",
                    "CRT2": "Fallacy Identification",
                    "CRT3": "Evidence-Based Decision Making",
                },
                "problemSolving": {
                    "PS1": "Risk Assessment",
                    "PS2": "Contingency Planning",
                    "PS3": "Corrective Action Implementation",
                },
                "decisionMaking": {
                    "DM1": "Option Identification",
                    "DM2": "Outcome Assessment",
                    "DM3": "Informed Choice Selection",
                },
                "strategicThinking": {
                    "ST1": "SWOT Analysis",
                    "ST2": "Long-Term Planning",
                    "ST3": "Future Trend Identification",
                },
                "emotionalIntelligence": {
                    "EI1": "Emotion Recognition and Management",
                    "EI2": "Relationship Building",
                    "EI3": "Empathetic Communication",
                },
            },
            "request": "adaptogen recommendation",
            "variables": {"adaptogen": "", "symptoms": ""},
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

    # Create columns for settings
    col1, col2 = st.columns(2)

    with col1:
        # Model selection
        available_models = [
            "models/gemini-2.0-pro-exp-02-0",
            "models/gemini-2.0-flash",
            "models/gemini-2.0-flash-thinking-exp-01-21",
            "models/learnlm-1.5-pro-experimental",
            "models/gemini-1.5-pro",
            "models/gemini-1.5-flash",
            "models/gemini-1.0-pro",
        ]
        selected_model = st.selectbox("Model", options=available_models, index=0)

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
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

    with col2:
        # Document context settings
        st.subheader("Document Context Settings")

        # Initialize contextual retrieval setting in session state if not present
        if "use_contextual_retrieval" not in st.session_state:
            st.session_state.use_contextual_retrieval = True

        # Toggle for contextual retrieval
        use_contextual = st.toggle(
            "Use Contextual Retrieval",
            value=st.session_state.use_contextual_retrieval,
            help="When enabled, the system will retrieve document chunks most relevant to your query",
        )

        # Retrieval method selection
        retrieval_methods = ["Hybrid (TF-IDF + BM25)", "TF-IDF Only", "BM25 Only"]
        retrieval_method = st.selectbox(
            "Retrieval Method",
            options=retrieval_methods,
            index=0,
            help="Method used to find relevant document chunks",
        )

        # Number of chunks to retrieve
        max_chunks = st.slider(
            "Max Document Chunks",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Maximum number of document chunks to include in context",
        )

    # Update settings button
    if st.button("Apply Settings"):
        # Update Gemini service
        if gemini_service.api_available and gemini_service.model is not None:
            try:
                gemini_service.model_name = selected_model
                # Create a new model instance with the selected model name
                gemini_service.model = genai.GenerativeModel(selected_model)

                # Update contextual retrieval settings
                st.session_state.use_contextual_retrieval = use_contextual
                st.session_state.retrieval_method = retrieval_method
                st.session_state.max_chunks = max_chunks

                if "context_manager" in st.session_state:
                    st.session_state.context_manager.use_contextual_retrieval = (
                        use_contextual
                    )
                    st.session_state.context_manager.document_context.use_contextual_chunks = (
                        use_contextual
                    )

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

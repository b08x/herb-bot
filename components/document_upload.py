"""
Document upload interface component for the Streamlit application.
"""
import os
from typing import Any, Dict, List, Optional

import streamlit as st
from services.document_service import document_service


def initialize_document_state():
    """Initialize the document state in the session state."""
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = {}

    if "selected_document_id" not in st.session_state:
        st.session_state.selected_document_id = None


def upload_document():
    """Handle document upload."""
    st.subheader("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=["pdf", "docx", "doc"],
        help="Upload a document to analyze",
    )

    if uploaded_file is not None:
        # Display file info in a more compact way
        st.caption(f"{uploaded_file.name} ({uploaded_file.size} bytes)")

        # Upload button - make it more compact
        if st.button("Process Document", use_container_width=True):
            with st.spinner("Processing document..."):
                try:
                    # Save the uploaded file
                    file_path = document_service.save_uploaded_file(uploaded_file)

                    # Process the document
                    document = document_service.process_document(file_path)

                    # Add to session state
                    st.session_state.uploaded_documents[document["id"]] = document

                    # Add to context manager if available
                    if "context_manager" in st.session_state:
                        st.session_state.context_manager.document_context.add_document(
                            document["id"], document["text"], document["metadata"]
                        )

                    st.success(f"Processed: {uploaded_file.name}")

                    # Set as selected document
                    st.session_state.selected_document_id = document["id"]

                except Exception as e:
                    st.error(f"Error: {e}")


def document_list():
    """Display the list of uploaded documents."""
    st.subheader("Uploaded Documents")

    if not st.session_state.uploaded_documents:
        st.info("No documents uploaded yet.")
        return

    # Get document summaries
    documents = []
    for doc_id, document in st.session_state.uploaded_documents.items():
        # Create a summary
        file_name = document["metadata"].get("file_name", "Unknown")
        file_type = document["metadata"].get("file_extension", "Unknown")
        file_size = document["metadata"].get("file_size_kb", 0)

        documents.append(
            {
                "id": doc_id,
                "name": file_name,
                "type": file_type,
                "size": f"{file_size} KB",
            }
        )

    # Display documents in a more compact list
    for i, doc in enumerate(documents):
        with st.container(border=True):
            # More compact display
            st.caption(f"{doc['name']} ({doc['type']}, {doc['size']})")

            # Create two columns for buttons
            col1, col2 = st.columns(2)

            with col1:
                # Select button
                if st.button(
                    "Select", key=f"select_{doc['id']}", use_container_width=True
                ):
                    st.session_state.selected_document_id = doc["id"]
                    st.rerun()

            with col2:
                # Delete button
                if st.button(
                    "Delete", key=f"delete_{doc['id']}", use_container_width=True
                ):
                    # Delete from document service
                    document_service.delete_document(doc["id"])

                    # Remove from session state
                    del st.session_state.uploaded_documents[doc["id"]]

                    # Remove from context manager if available
                    if "context_manager" in st.session_state:
                        st.session_state.context_manager.document_context.remove_document(
                            doc["id"]
                        )

                    # Clear selected document if it was this one
                    if st.session_state.selected_document_id == doc["id"]:
                        st.session_state.selected_document_id = None

                    st.success(f"Deleted: {doc['name']}")
                    st.rerun()


def document_viewer():
    """Display the selected document."""
    st.subheader("Document Viewer")

    if st.session_state.selected_document_id is None:
        st.info("Select a document to view.")
        return

    # Get the selected document
    doc_id = st.session_state.selected_document_id
    document = st.session_state.uploaded_documents.get(doc_id)

    if document is None:
        st.error("Document not found.")
        return

    # Display document info in a more compact way
    file_name = document["metadata"].get("file_name", "Unknown")
    file_type = document["metadata"].get("file_extension", "Unknown")
    file_size = document["metadata"].get("file_size_kb", 0)

    st.caption(f"**{file_name}** ({file_type}, {file_size} KB)")

    # Display document metadata in a collapsible section
    with st.expander("Metadata"):
        st.json(document["metadata"])

    # Display document content in a more compact way
    with st.expander("Content", expanded=True):
        # Create a smaller text area with the document content
        text_content = document["text"]
        st.text_area(
            "",
            value=text_content,
            height=200,
            disabled=True,
            label_visibility="collapsed",
        )

    # Add to chat context button - make it more prominent
    if st.button("Use in Chat", use_container_width=True, type="primary"):
        # Add to context manager if available
        if "context_manager" in st.session_state:
            st.session_state.context_manager.document_context.add_document(
                document["id"], document["text"], document["metadata"]
            )

            st.success(f"Added to chat: {file_name}")
        else:
            st.error("Chat context not available.")


def document_search():
    """Search within uploaded documents using contextual retrieval."""
    st.subheader("Document Search")

    if not st.session_state.uploaded_documents:
        st.info("No documents uploaded yet.")
        return

    # Search input
    search_query = st.text_input("Search in documents")

    # Search method as a radio button to save space
    search_method = st.radio(
        "Method",
        options=["Hybrid", "TF-IDF", "BM25", "Basic"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if search_query and len(search_query) >= 3:
        with st.spinner("Searching..."):
            # Map the selected method to the API parameter
            method_map = {
                "Hybrid": "hybrid",
                "TF-IDF": "tfidf",
                "BM25": "bm25",
                "Basic": "basic",
            }
            method = method_map[search_method]

            # Use basic search or contextual search based on selection
            if method == "basic":
                # Basic search (original implementation)
                results = []

                for doc_id, document in st.session_state.uploaded_documents.items():
                    if search_query.lower() in document["text"].lower():
                        # Get context around the match
                        text = document["text"]
                        query_pos = text.lower().find(search_query.lower())

                        # Get context (100 characters before and after)
                        start = max(0, query_pos - 100)
                        end = min(len(text), query_pos + len(search_query) + 100)
                        context = text[start:end]

                        # Add ellipsis if context is truncated
                        if start > 0:
                            context = "..." + context
                        if end < len(text):
                            context = context + "..."

                        # Highlight the match
                        highlighted_context = context.replace(
                            search_query, f"**{search_query}**"
                        )

                        results.append(
                            {
                                "id": doc_id,
                                "name": document["metadata"].get(
                                    "file_name", "Unknown"
                                ),
                                "context": highlighted_context,
                                "score": 1.0,  # Default score for basic search
                            }
                        )
            else:
                # Use contextual search with the document service
                results = document_service.contextual_search(
                    query=search_query, top_k=3, method=method
                )

                # Format results for display
                for result in results:
                    # Highlight the query terms if possible
                    context = result["context"]
                    # Simple highlighting - this could be improved with regex
                    for term in search_query.lower().split():
                        if len(term) > 3:  # Only highlight terms with more than 3 chars
                            context = context.replace(term, f"**{term}**")

                    result["context"] = context

        # Display results
        if results:
            st.caption(f"Found {len(results)} matches")

            for i, result in enumerate(results):
                with st.container(border=True):
                    # Display document name and score in a more compact way
                    score = result.get("score", 0)
                    st.caption(
                        f"**{result.get('name', 'Unknown')}** (Score: {score:.2f})"
                    )

                    # Display context in a more compact way
                    st.markdown(result["context"])

                    # Create columns for buttons
                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        # View document button
                        if st.button(
                            "View",
                            key=f"view_{i}_{result['id']}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_document_id = result["id"]
                            st.rerun()

                    with btn_col2:
                        # Add to chat context button
                        if st.button(
                            "Add",
                            key=f"add_{i}_{result['id']}",
                            use_container_width=True,
                        ):
                            if "context_manager" in st.session_state:
                                # Get the full document
                                doc = st.session_state.uploaded_documents.get(
                                    result["id"]
                                )
                                if doc:
                                    # Add to context manager
                                    st.session_state.context_manager.document_context.add_document(
                                        doc["id"], doc["text"], doc["metadata"]
                                    )
                                    st.success(
                                        f"Added: {doc['metadata'].get('file_name', 'Document')}"
                                    )
                            else:
                                st.error("Chat context not available.")
        else:
            st.info(f"No matches for '{search_query}'")


def render_document_upload_tab():
    """Render the document upload tab in the Streamlit application."""
    # Initialize document state
    initialize_document_state()

    # Upload section
    upload_document()

    st.divider()

    # Document list
    document_list()

    st.divider()

    # Document viewer
    document_viewer()

    st.divider()

    # Document search
    document_search()

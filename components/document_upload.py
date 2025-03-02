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
        # Display file info
        st.write(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")

        # Upload button
        if st.button("Process Document"):
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

                    st.success(f"Document processed successfully: {uploaded_file.name}")

                    # Set as selected document
                    st.session_state.selected_document_id = document["id"]

                except Exception as e:
                    st.error(f"Error processing document: {e}")


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

    # Create columns for each document
    cols = st.columns(min(3, len(documents)))

    for i, doc in enumerate(documents):
        with cols[i % len(cols)]:
            # Create a card-like display
            with st.container(border=True):
                st.write(f"**{doc['name']}**")
                st.write(f"Type: {doc['type']}")
                st.write(f"Size: {doc['size']}")

                # Select button
                if st.button("Select", key=f"select_{doc['id']}"):
                    st.session_state.selected_document_id = doc["id"]
                    st.rerun()

                # Delete button
                if st.button("Delete", key=f"delete_{doc['id']}"):
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

                    st.success(f"Document deleted: {doc['name']}")
                    st.rerun()


def document_viewer():
    """Display the selected document."""
    st.subheader("Document Viewer")

    if st.session_state.selected_document_id is None:
        st.info("Select a document to view its contents.")
        return

    # Get the selected document
    doc_id = st.session_state.selected_document_id
    document = st.session_state.uploaded_documents.get(doc_id)

    if document is None:
        st.error("Selected document not found.")
        return

    # Display document info
    file_name = document["metadata"].get("file_name", "Unknown")
    file_type = document["metadata"].get("file_extension", "Unknown")
    file_size = document["metadata"].get("file_size_kb", 0)

    st.write(f"**{file_name}**")
    st.write(f"Type: {file_type} | Size: {file_size} KB")

    # Display document metadata
    with st.expander("Document Metadata"):
        st.json(document["metadata"])

    # Display document content
    st.markdown("### Document Content")

    # Create a text area with the document content
    text_content = document["text"]
    st.text_area("Document Text", value=text_content, height=400, disabled=True)

    # Add to chat context button
    if st.button("Use in Chat"):
        # Add to context manager if available
        if "context_manager" in st.session_state:
            st.session_state.context_manager.document_context.add_document(
                document["id"], document["text"], document["metadata"]
            )

            st.success(f"Document added to chat context: {file_name}")
        else:
            st.error("Chat context not available.")


def document_search():
    """Search within uploaded documents."""
    st.subheader("Document Search")

    if not st.session_state.uploaded_documents:
        st.info("No documents uploaded yet.")
        return

    # Search input
    search_query = st.text_input("Search in documents")

    if search_query and len(search_query) >= 3:
        # Search in documents
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
                        "name": document["metadata"].get("file_name", "Unknown"),
                        "context": highlighted_context,
                    }
                )

        # Display results
        if results:
            st.write(f"Found {len(results)} matches:")

            for result in results:
                with st.container(border=True):
                    st.write(f"**{result['name']}**")
                    st.markdown(result["context"])

                    # View document button
                    if st.button("View Document", key=f"view_{result['id']}"):
                        st.session_state.selected_document_id = result["id"]
                        st.rerun()
        else:
            st.info(f"No matches found for '{search_query}'")


def render_document_upload_tab():
    """Render the document upload tab in the Streamlit application."""
    st.title("Document Management")

    # Initialize document state
    initialize_document_state()

    # Create two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        # Upload section
        upload_document()

        st.divider()

        # Document list
        document_list()

    with col2:
        # Document viewer
        document_viewer()

        st.divider()

        # Document search
        document_search()

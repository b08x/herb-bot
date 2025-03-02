"""
Service for processing and managing documents.
"""
import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from utils.text_extraction import extract_text_from_file, get_file_metadata


class DocumentService:
    """Service for processing and managing documents."""

    def __init__(self, upload_dir: str = "data/uploads"):
        """
        Initialize the document service.

        Args:
            upload_dir: Directory for storing uploaded files
        """
        self.upload_dir = upload_dir
        self.documents = {}

        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_uploaded_file(self, file_obj, filename: Optional[str] = None) -> str:
        """
        Save an uploaded file to the upload directory.

        Args:
            file_obj: File object from Streamlit
            filename: Optional filename to use (if None, use the original filename)

        Returns:
            Path to the saved file
        """
        # Generate a unique filename if not provided
        if filename is None:
            filename = file_obj.name

        # Add timestamp to filename to make it unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, extension = os.path.splitext(filename)
        unique_filename = f"{base_name}_{timestamp}{extension}"

        # Create the file path
        file_path = os.path.join(self.upload_dir, unique_filename)

        # Save the file
        with open(file_path, "wb") as f:
            f.write(file_obj.getbuffer())

        return file_path

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and extract its text and metadata.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with document information
        """
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())

        # Get file metadata
        metadata = get_file_metadata(file_path)

        # Extract text from the file
        text, extraction_metadata = extract_text_from_file(file_path)

        # Combine metadata
        metadata.update(extraction_metadata)

        # Create document record
        document = {
            "id": doc_id,
            "file_path": file_path,
            "text": text,
            "metadata": metadata,
            "processed_at": datetime.now().isoformat(),
        }

        # Store document
        self.documents[doc_id] = document

        return document

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document dictionary or None if not found
        """
        return self.documents.get(doc_id)

    def get_all_documents(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all documents.

        Returns:
            Dictionary mapping document IDs to document dictionaries
        """
        return self.documents

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document.

        Args:
            doc_id: Document ID

        Returns:
            True if document was deleted, False if not found
        """
        if doc_id in self.documents:
            # Get the file path
            file_path = self.documents[doc_id]["file_path"]

            # Delete the file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)

            # Remove from documents dictionary
            del self.documents[doc_id]

            return True

        return False

    def get_document_text(self, doc_id: str) -> Optional[str]:
        """
        Get the text content of a document.

        Args:
            doc_id: Document ID

        Returns:
            Document text or None if not found
        """
        document = self.get_document(doc_id)
        if document:
            return document["text"]
        return None

    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a document (metadata and preview).

        Args:
            doc_id: Document ID

        Returns:
            Document summary dictionary or None if not found
        """
        document = self.get_document(doc_id)
        if document:
            # Create a preview of the text (first 200 characters)
            text_preview = (
                document["text"][:200] + "..."
                if len(document["text"]) > 200
                else document["text"]
            )

            return {
                "id": document["id"],
                "file_name": document["metadata"].get("file_name", "Unknown"),
                "file_type": document["metadata"].get("file_extension", "Unknown"),
                "file_size_kb": document["metadata"].get("file_size_kb", 0),
                "processed_at": document["processed_at"],
                "text_preview": text_preview,
            }

        return None

    def get_all_document_summaries(self) -> List[Dict[str, Any]]:
        """
        Get summaries of all documents.

        Returns:
            List of document summary dictionaries
        """
        return [self.get_document_summary(doc_id) for doc_id in self.documents]

    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for documents containing the query.

        Args:
            query: Search query

        Returns:
            List of matching document dictionaries
        """
        results = []

        for doc_id, document in self.documents.items():
            if query.lower() in document["text"].lower():
                # Create a result with context
                text = document["text"]
                query_pos = text.lower().find(query.lower())

                # Get context around the match (100 characters before and after)
                start = max(0, query_pos - 100)
                end = min(len(text), query_pos + len(query) + 100)
                context = text[start:end]

                # Add ellipsis if context is truncated
                if start > 0:
                    context = "..." + context
                if end < len(text):
                    context = context + "..."

                results.append(
                    {
                        "id": doc_id,
                        "file_name": document["metadata"].get("file_name", "Unknown"),
                        "context": context,
                        "match_position": query_pos,
                    }
                )

        return results

    def export_document_data(self, doc_id: str, output_path: str) -> bool:
        """
        Export document data to a JSON file.

        Args:
            doc_id: Document ID
            output_path: Path to save the JSON file

        Returns:
            True if export was successful, False otherwise
        """
        document = self.get_document(doc_id)
        if document:
            try:
                with open(output_path, "w") as f:
                    json.dump(document, f, indent=2)
                return True
            except Exception as e:
                print(f"Error exporting document: {e}")

        return False


# Create a singleton instance
document_service = DocumentService()

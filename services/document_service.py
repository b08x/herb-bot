"""
Service for processing and managing documents.
"""
import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from utils.text_chunking import (BM25Retriever, HybridRetriever, TextChunker,
                                 TFIDFRetriever, get_contextual_chunks)
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

        # Initialize text chunker
        self.text_chunker = TextChunker(
            chunk_size=1000, chunk_overlap=200, split_by="paragraph"
        )

        # Initialize retrievers
        self.tfidf_retriever = TFIDFRetriever()
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever()

        # Flag to track if retrievers are initialized with documents
        self.retrievers_initialized = False

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

        # Chunk the document text
        chunked_document = self.text_chunker.chunk_document(document)

        # Store document with chunks
        self.documents[doc_id] = chunked_document

        # Reset retrievers initialization flag
        self.retrievers_initialized = False

        return chunked_document

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

    def _initialize_retrievers(self):
        """Initialize retrievers with all documents if not already initialized."""
        if not self.retrievers_initialized:
            # Clear existing documents in retrievers
            self.tfidf_retriever = TFIDFRetriever()
            self.bm25_retriever = BM25Retriever()
            self.hybrid_retriever = HybridRetriever()

            # Add all documents to retrievers
            for doc_id, document in self.documents.items():
                # Skip documents without chunks
                if "chunks" not in document:
                    continue

                self.tfidf_retriever.add_document(document)
                self.bm25_retriever.add_document(document)
                self.hybrid_retriever.add_document(document)

            self.retrievers_initialized = True

    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for documents containing the query.

        This method uses a hybrid of TF-IDF and BM25 for contextual retrieval.

        Args:
            query: Search query

        Returns:
            List of matching document dictionaries
        """
        # If no documents, return empty list
        if not self.documents:
            return []

        # Use contextual search if documents have chunks
        if any("chunks" in doc for doc in self.documents.values()):
            return self.contextual_search(query)

        # Fallback to basic search if no chunks
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
                        "score": 1.0,  # Default score for basic search
                    }
                )

        return results

    def contextual_search(
        self, query: str, top_k: int = 5, method: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Search for document chunks relevant to the query using contextual retrieval.

        Args:
            query: Search query
            top_k: Number of top results to return
            method: Retrieval method ('tfidf', 'bm25', or 'hybrid')

        Returns:
            List of relevant chunks with scores and document information
        """
        # Initialize retrievers if needed
        self._initialize_retrievers()

        # Get document list
        documents = list(self.documents.values())

        # If no documents, return empty list
        if not documents:
            return []

        # Get retriever based on method
        if method == "tfidf":
            retriever = self.tfidf_retriever
        elif method == "bm25":
            retriever = self.bm25_retriever
        else:  # hybrid
            retriever = self.hybrid_retriever

        # Search for relevant chunks
        chunk_results = retriever.search(query, top_k=top_k)

        # Format results
        results = []
        for chunk in chunk_results:
            doc_id = chunk["doc_id"]
            document = self.documents.get(doc_id)

            if document:
                result = {
                    "id": doc_id,
                    "chunk_index": chunk["index"],
                    "file_name": document["metadata"].get("file_name", "Unknown"),
                    "context": chunk["text"],
                    "score": chunk.get("score", 0.0),
                }

                # Add combined score if available
                if "combined_score" in chunk:
                    result["combined_score"] = chunk["combined_score"]
                    result["tfidf_score"] = chunk.get("tfidf_score", 0.0)
                    result["bm25_score"] = chunk.get("bm25_score", 0.0)

                results.append(result)

        return results

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search on documents using contextual chunks.

        This is a wrapper around contextual_search that uses the hybrid method.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of relevant chunks with scores and document information
        """
        return self.contextual_search(query, top_k=top_k, method="hybrid")

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

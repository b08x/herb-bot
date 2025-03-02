"""
Utility functions for managing conversation context and document integration.
"""
import json
from typing import Any, Dict, List, Optional, Union

from utils.text_chunking import TextChunker, get_contextual_chunks


class Message:
    """Class representing a chat message."""

    def __init__(self, role: str, content: str):
        """
        Initialize a message.

        Args:
            role: The role of the message sender (user, assistant, system)
            content: The content of the message
        """
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        """Convert message to dictionary format."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Message":
        """Create a message from dictionary data."""
        return cls(role=data["role"], content=data["content"])


class Conversation:
    """Class for managing conversation history and context."""

    def __init__(self, system_prompt: Optional[Dict[str, Any]] = None):
        """
        Initialize a conversation.

        Args:
            system_prompt: Optional system prompt to initialize the conversation
        """
        self.messages: List[Message] = []

        # Add system prompt if provided
        if system_prompt:
            if isinstance(system_prompt, dict):
                system_content = json.dumps(system_prompt)
            else:
                system_content = str(system_prompt)

            self.messages.append(Message("system", system_content))

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.

        Args:
            content: The content of the user message
        """
        self.messages.append(Message("user", content))

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation.

        Args:
            content: The content of the assistant message
        """
        self.messages.append(Message("assistant", content))

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in the conversation.

        Returns:
            List of message dictionaries
        """
        return [message.to_dict() for message in self.messages]

    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        """
        Get the last n messages in the conversation.

        Args:
            n: Number of messages to retrieve

        Returns:
            List of the last n message dictionaries
        """
        return [message.to_dict() for message in self.messages[-n:]]

    def clear(self) -> None:
        """Clear all messages except the system prompt."""
        if self.messages and self.messages[0].role == "system":
            system_prompt = self.messages[0]
            self.messages = [system_prompt]
        else:
            self.messages = []


class DocumentContext:
    """Class for managing document context."""

    def __init__(self):
        """Initialize document context."""
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.text_chunker = TextChunker(
            chunk_size=1000, chunk_overlap=200, split_by="paragraph"
        )
        self.use_contextual_chunks = True

    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """
        Add a document to the context.

        Args:
            doc_id: Unique identifier for the document
            content: Text content of the document
            metadata: Document metadata
        """
        document = {"content": content, "metadata": metadata}

        # Create chunks if enabled
        if self.use_contextual_chunks:
            # Create a document in the format expected by the chunker
            doc_for_chunking = {"id": doc_id, "text": content, "metadata": metadata}

            # Chunk the document
            chunked_doc = self.text_chunker.chunk_document(doc_for_chunking)

            # Add chunks to the document
            document["chunks"] = chunked_doc.get("chunks", [])

        self.documents[doc_id] = document

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document from the context.

        Args:
            doc_id: Document identifier

        Returns:
            Document data or None if not found
        """
        return self.documents.get(doc_id)

    def get_all_documents(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all documents in the context.

        Returns:
            Dictionary of all documents
        """
        return self.documents

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the context.

        Args:
            doc_id: Document identifier

        Returns:
            True if document was removed, False if not found
        """
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all documents from the context."""
        self.documents = {}

    def get_contextual_chunks(
        self, query: str, top_k: int = 5, method: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Get contextual chunks from documents based on a query.

        Args:
            query: Query to find relevant chunks
            top_k: Number of top chunks to return
            method: Retrieval method ('hybrid', 'tfidf', or 'bm25')

        Returns:
            List of relevant chunks with scores
        """
        # Convert documents to the format expected by get_contextual_chunks
        docs_for_retrieval = []
        for doc_id, doc_data in self.documents.items():
            # Skip documents without chunks
            if "chunks" not in doc_data:
                continue

            # Create a document with the expected format
            doc = {
                "id": doc_id,
                "text": doc_data["content"],
                "metadata": doc_data["metadata"],
                "chunks": doc_data["chunks"],
            }

            docs_for_retrieval.append(doc)

        # Get contextual chunks
        return get_contextual_chunks(
            query, docs_for_retrieval, top_k=top_k, method=method
        )

    def get_combined_content(
        self,
        max_length: Optional[int] = None,
        query: Optional[str] = None,
        method: str = "hybrid",
        max_chunks: int = 5,
    ) -> str:
        """
        Get combined content of all documents.

        If a query is provided and contextual chunks are enabled, returns relevant chunks.
        Otherwise, returns all document content.

        Args:
            max_length: Optional maximum length of the combined content
            query: Optional query to find relevant chunks
            method: Retrieval method ('hybrid', 'tfidf', or 'bm25')
            max_chunks: Maximum number of chunks to retrieve

        Returns:
            Combined document content
        """
        # If query is provided and contextual chunks are enabled, use contextual retrieval
        if query and self.use_contextual_chunks:
            chunks = self.get_contextual_chunks(query, top_k=max_chunks, method=method)

            if chunks:
                combined = ""

                for chunk in chunks:
                    doc_id = chunk.get("doc_id", "unknown")
                    doc_data = self.documents.get(doc_id, {})
                    doc_name = doc_data.get("metadata", {}).get("file_name", doc_id)

                    combined += f"\n\n--- Document: {doc_name} (Relevance: {chunk.get('score', 0):.2f}) ---\n\n"
                    combined += chunk.get("text", "")

                if max_length and len(combined) > max_length:
                    return combined[:max_length] + "..."

                return combined

        # Default behavior: combine all documents
        combined = ""

        for doc_id, doc_data in self.documents.items():
            doc_content = doc_data["content"]
            doc_name = doc_data["metadata"].get("file_name", doc_id)

            combined += f"\n\n--- Document: {doc_name} ---\n\n"
            combined += doc_content

        if max_length and len(combined) > max_length:
            return combined[:max_length] + "..."

        return combined


class ContextManager:
    """Class for managing both conversation and document context."""

    def __init__(self, system_prompt: Optional[Dict[str, Any]] = None):
        """
        Initialize context manager.

        Args:
            system_prompt: Optional system prompt for the conversation
        """
        self.conversation = Conversation(system_prompt)
        self.document_context = DocumentContext()
        self.use_contextual_retrieval = True

    def prepare_context_for_model(
        self,
        include_docs: bool = True,
        max_doc_length: Optional[int] = 8000,
        max_messages: Optional[int] = None,
        query: Optional[str] = None,
        method: str = "hybrid",
        max_chunks: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Prepare context for sending to the model.

        Args:
            include_docs: Whether to include document content
            max_doc_length: Maximum length of document content to include
            max_messages: Maximum number of messages to include
            query: Optional query for contextual document retrieval
            method: Retrieval method ('hybrid', 'tfidf', or 'bm25')
            max_chunks: Maximum number of chunks to retrieve

        Returns:
            List of message dictionaries ready for the model
        """
        messages = self.conversation.get_messages()

        if max_messages:
            # Always keep system message if present
            if messages and messages[0]["role"] == "system":
                messages = [messages[0]] + messages[-(max_messages - 1) :]
            else:
                messages = messages[-max_messages:]

        if include_docs and self.document_context.documents:
            # Use the query for contextual retrieval if available and enabled
            if query and self.use_contextual_retrieval:
                doc_content = self.document_context.get_combined_content(
                    max_length=max_doc_length,
                    query=query,
                    method=method,
                    max_chunks=max_chunks,
                )
            else:
                doc_content = self.document_context.get_combined_content(
                    max_length=max_doc_length
                )

            # Find the right position to insert document context
            # If there's a system message, insert after it
            if messages and messages[0]["role"] == "system":
                # Append to system message
                messages[0]["content"] += f"\n\nReference Documents:\n{doc_content}"
            else:
                # Insert as a new system message at the beginning
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": f"Reference Documents:\n{doc_content}",
                    },
                )

        return messages

    def get_relevant_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get contextually relevant chunks from documents based on a query.

        Args:
            query: Query to find relevant chunks
            top_k: Number of top chunks to return

        Returns:
            List of relevant chunks with scores
        """
        return self.document_context.get_contextual_chunks(query, top_k=top_k)

"""
Utility functions for managing conversation context and document integration.
"""
import json
from typing import Any, Dict, List, Optional


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

    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """
        Add a document to the context.

        Args:
            doc_id: Unique identifier for the document
            content: Text content of the document
            metadata: Document metadata
        """
        self.documents[doc_id] = {"content": content, "metadata": metadata}

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

    def get_combined_content(self, max_length: Optional[int] = None) -> str:
        """
        Get combined content of all documents.

        Args:
            max_length: Optional maximum length of the combined content

        Returns:
            Combined document content
        """
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

    def prepare_context_for_model(
        self,
        include_docs: bool = True,
        max_doc_length: Optional[int] = 8000,
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Prepare context for sending to the model.

        Args:
            include_docs: Whether to include document content
            max_doc_length: Maximum length of document content to include
            max_messages: Maximum number of messages to include

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

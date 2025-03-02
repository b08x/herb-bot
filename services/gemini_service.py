"""
Service for interacting with Google's Gemini API.
"""
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Flag to track if Gemini API is available
GEMINI_API_AVAILABLE = False

try:
    import google.generativeai as genai

    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_API_AVAILABLE = True
    else:
        print("GEMINI_API_KEY environment variable is not set")
except ImportError as e:
    print(f"Google Generative AI package not available: {e}")
    print("Chat functionality will be limited.")


class GeminiService:
    """Service for interacting with Google's Gemini API."""

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the Gemini service.

        Args:
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        self.model = None
        self.system_prompt = None
        self.api_available = GEMINI_API_AVAILABLE

        if self.api_available:
            try:
                self.model = genai.GenerativeModel(model_name)
            except Exception as e:
                print(f"Error initializing Gemini model: {e}")
                self.api_available = False

    def set_system_prompt(self, system_prompt: Dict[str, Any]) -> None:
        """
        Set the system prompt for the model.

        Args:
            system_prompt: System prompt as a dictionary
        """
        self.system_prompt = system_prompt

    def get_system_prompt_string(self) -> Optional[str]:
        """
        Get the system prompt as a string.

        Returns:
            System prompt as a string or None if not set
        """
        if self.system_prompt:
            return json.dumps(self.system_prompt)
        return None

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Temperature parameter for generation
            max_output_tokens: Maximum number of tokens to generate

        Returns:
            Generated response as a string
        """
        # Check if API is available
        if not self.api_available or self.model is None:
            return (
                "The Gemini API is not available. Please check your API key and ensure "
                "the google-generativeai package is installed correctly."
            )

        try:
            # Create a chat session
            chat = self.model.start_chat(history=[])

            # Add messages to the chat
            for message in messages:
                if message["role"] == "user":
                    chat.send_message(message["content"])
                elif message["role"] == "assistant":
                    # We can't directly add assistant messages to the chat history
                    # So we'll simulate it by adding it to the history
                    chat._history.append(
                        {"role": "model", "parts": [{"text": message["content"]}]}
                    )
                elif message["role"] == "system" and message == messages[0]:
                    # System message is handled differently
                    # For newer versions of the API, system_instruction might not be supported
                    try:
                        # Try with system_instruction parameter
                        chat = self.model.start_chat(
                            history=[], system_instruction=message["content"]
                        )
                    except TypeError as e:
                        # If system_instruction is not supported, add as a regular message
                        print(f"System instruction not supported: {e}")
                        # Create a new chat
                        chat = self.model.start_chat(history=[])
                        # Add system message as a user message (workaround)
                        chat.send_message(f"System instructions: {message['content']}")

            # Generate response
            response = chat.send_message(
                messages[-1]["content"] if messages[-1]["role"] == "user" else "",
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )

            return response.text

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}\n\nPlease check your API key and connection."

    def generate_response_with_prompt(
        self,
        user_input: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
    ) -> str:
        """
        Generate a response using the system prompt and user input.

        Args:
            user_input: User input text
            context: Optional additional context
            temperature: Temperature parameter for generation
            max_output_tokens: Maximum number of tokens to generate

        Returns:
            Generated response as a string
        """
        messages = []

        # Add system prompt if available
        if self.system_prompt:
            system_content = json.dumps(self.system_prompt)
            messages.append({"role": "system", "content": system_content})

        # Add context if available
        if context:
            if messages and messages[0]["role"] == "system":
                # Append to system message
                messages[0]["content"] += f"\n\nContext:\n{context}"
            else:
                # Add as system message
                messages.append({"role": "system", "content": f"Context:\n{context}"})

        # Add user input
        messages.append({"role": "user", "content": user_input})

        # Generate response
        return self.generate_response(messages, temperature, max_output_tokens)

    def list_available_models(self) -> List[str]:
        """
        List available Gemini models.

        Returns:
            List of available model names
        """
        try:
            models = genai.list_models()
            return [model.name for model in models if "gemini" in model.name]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []


# Create a singleton instance
gemini_service = GeminiService()

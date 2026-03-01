#!/usr/bin/env python3
"""
Inception Labs Mercury-2 Service

Direct API client for Mercury-2, avoiding LangChain complications.
Supports both instant and default reasoning modes.
"""

import os
import requests
import asyncio
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class _Message:
    """Simple message class to mimic LangChain's message format."""
    def __init__(self, content: str):
        self.content = content

INCEPTION_API_KEY = os.getenv("INCEPTION_API_KEY")
INCEPTION_ENDPOINT = "https://api.inceptionlabs.ai/v1/chat/completions"


class Mercury2Client:
    """Direct client for Inception Labs Mercury-2 API."""

    def __init__(self, model: str = "mercury-2"):
        """
        Initialize Mercury-2 client.

        Args:
            model: Model name (default: "mercury-2")
                   Use "mercury-2-instant" for instant mode
        """
        self.api_key = INCEPTION_API_KEY
        self.endpoint = INCEPTION_ENDPOINT
        self.model = "mercury-2"  # Always use mercury-2 as the API model name
        self.is_instant = model.endswith("-instant")

        if self.is_instant:
            logger.info("🚀 Mercury-2 Instant mode enabled (fast ~0.8s)")
        else:
            logger.info("🧠 Mercury-2 Default mode enabled (deep ~2-3s)")

    def invoke(self, messages, **kwargs) -> str:
        """
        Send a chat completion request to Mercury-2.

        Args:
            messages: List of messages (LangChain HumanMessage/AIMessage or dicts)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            The response content as a string
        """
        # Convert LangChain messages to dicts if needed
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append(msg)
            elif hasattr(msg, 'content'):
                # LangChain message object
                role = getattr(msg, 'role', 'user')
                if hasattr(msg, 'type'):
                    # Handle SystemMessage, HumanMessage, AIMessage
                    msg_type = getattr(msg, 'type', 'user')
                    if msg_type == 'system':
                        role = 'system'
                    elif msg_type == 'human':
                        role = 'user'
                    elif msg_type == 'ai':
                        role = 'assistant'
                formatted_messages.append({
                    "role": role,
                    "content": getattr(msg, 'content', str(msg))
                })
            else:
                # Fallback for string messages
                formatted_messages.append({"role": "user", "content": str(msg)})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        body = {
            "model": self.model,
            "messages": formatted_messages
        }

        # Add reasoning_effort for instant mode
        if self.is_instant:
            body["reasoning_effort"] = "instant"
        else:
            # Default mode uses max_tokens if specified
            if "max_tokens" in kwargs:
                body["max_tokens"] = kwargs["max_tokens"]

        # Add any additional parameters
        body.update(kwargs)

        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=body,
                timeout=120
            )
            response.raise_for_status()

            data = response.json()

            # Log token usage if available
            usage = data.get("usage", {})
            reasoning_tokens = usage.get("reasoning_tokens")
            if reasoning_tokens is not None:
                logger.debug(f"Mercury-2 reasoning_tokens: {reasoning_tokens}")

            # Extract and return the content
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                # Return a response-like object with content attribute for compatibility
                return _Message(content)
            else:
                raise ValueError("No choices in response")

        except requests.exceptions.RequestException as e:
            logger.error(f"Mercury-2 API error: {e}")
            raise

    async def ainvoke(self, messages, **kwargs) -> str:
        """
        Async version of invoke for compatibility with LangChain patterns.

        Args:
            messages: List of messages (LangChain HumanMessage/AIMessage or dicts)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            The response content as a string
        """
        # Run the synchronous invoke in an executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, messages, **kwargs)


def create_mercury_llm(model: str) -> Mercury2Client:
    """
    Factory function to create a Mercury-2 client.

    Compatible with the existing LLM initialization pattern.
    """
    return Mercury2Client(model=model)

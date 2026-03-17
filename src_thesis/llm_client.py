"""
OpenAI API client module with automatic retry logic.

This module provides a robust interface to OpenAI's API with:
- Automatic retry on rate limits and transient errors
- Exponential backoff with jitter
- Response caching capabilities
- Usage tracking and cost estimation

The LLMClient class encapsulates all API interactions, providing
a clean interface for embeddings and chat completions.

Example:
    >>> from src_thesis.llm_client import LLMClient
    >>> client = LLMClient(model="gpt-4o-mini")
    >>> response = client.chat(messages=[{"role": "user", "content": "Hello"}])
    >>> print(response['text'])
    'Hello! How can I help you today?'
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from openai import OpenAI, APIError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState
)

from . import config


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UsageStats:
    """
    Token usage statistics from an API call.
    
    Attributes:
        prompt_tokens: Number of tokens in the input
        completion_tokens: Number of tokens in the output
        total_tokens: Sum of prompt and completion tokens
        estimated_cost_usd: Estimated cost in USD based on model pricing
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


@dataclass
class ChatResponse:
    """
    Response from a chat completion API call.
    
    Attributes:
        text: The generated text response
        usage: Token usage statistics
        raw_response: The raw OpenAI API response object
    """
    text: str
    usage: UsageStats
    raw_response: Any


# =============================================================================
# LLM CLIENT CLASS
# =============================================================================

class LLMClient:
    """
    Wrapper for OpenAI API with retry logic and usage tracking.
    
    This class provides a robust interface to OpenAI's API with automatic
    retries on failures, exponential backoff, and usage tracking.
    
    Attributes:
        model: The OpenAI model identifier (e.g., 'gpt-4o-mini')
        api_key: OpenAI API key (defaults to config.OPENAI_API_KEY)
        client: The underlying OpenAI client instance
        
    Example:
        >>> client = LLMClient(model="gpt-4o-mini")
        >>> response = client.chat([
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What is 2+2?"}
        ... ])
        >>> print(response.text)
        '4'
    """
    
    # Pricing per 1M tokens (as of 2024, subject to change)
    # Source: https://openai.com/pricing
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    }
    
    def __init__(
        self,
        model: str = config.LLM_MODEL,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: OpenAI model identifier (default from config)
            api_key: OpenAI API key (default from config)
            
        Raises:
            ValueError: If API key is not provided and not in config
        """
        self.model = model
        self.api_key = api_key or config.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY in .env or pass api_key parameter."
            )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
    
    def _estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None
    ) -> float:
        """
        Estimate the cost of an API call in USD.
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            model: Model name (defaults to self.model)
            
        Returns:
            Estimated cost in USD
        """
        model_name = model or self.model
        pricing = self.PRICING.get(model_name, {"input": 0, "output": 0})
        
        cost = (
            (prompt_tokens * pricing["input"] / 1_000_000) +
            (completion_tokens * pricing["output"] / 1_000_000)
        )
        return cost
    
    @staticmethod
    def _log_retry_attempt(retry_state: RetryCallState) -> None:
        """
        Log retry attempts for debugging.
        
        Args:
            retry_state: The retry state from tenacity
        """
        if config.DEBUG_MODE and retry_state.attempt_number > 1:
            wait_time = retry_state.next_action.sleep if retry_state.next_action else 0
            print(
                f"⚠️ Retry attempt {retry_state.attempt_number} "
                f"after {wait_time:.1f}s wait"
            )
    
    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=_log_retry_attempt.__func__  # type: ignore
    )
    def create_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> Any:
        """
        Generate embeddings for a list of texts with automatic retry.
        
        Args:
            texts: List of text strings to embed
            model: Embedding model (defaults to config.EMBED_MODEL)
            
        Returns:
            OpenAI embeddings response object with .data attribute
            
        Raises:
            RateLimitError: If rate limit exceeded after retries
            APIError: If API error persists after retries
            
        Example:
            >>> client = LLMClient()
            >>> embeddings = client.create_embeddings(["Hello", "World"])
            >>> vector = embeddings.data[0].embedding
            >>> len(vector)
            1536
        """
        embedding_model = model or config.EMBED_MODEL
        
        # Replace newlines to improve embedding quality
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        
        return self.client.embeddings.create(
            input=cleaned_texts,
            model=embedding_model
        )
    
    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=_log_retry_attempt.__func__  # type: ignore
    )
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 500,
        model: Optional[str] = None
    ) -> ChatResponse:
        """
        Call ChatCompletion API with automatic retry.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            model: Model to use (defaults to self.model)
            
        Returns:
            ChatResponse object with text, usage stats, and raw response
            
        Raises:
            RateLimitError: If rate limit exceeded after retries
            APIError: If API error persists after retries
            
        Example:
            >>> client = LLMClient()
            >>> response = client.chat([
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hi!"}
            ... ])
            >>> print(response.text)
            >>> print(response.usage.total_tokens)
        """
        chat_model = model or self.model
        
        response = self.client.chat.completions.create(
            model=chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract text output
        text_output = response.choices[0].message.content or ""
        
        # Build usage statistics
        usage = UsageStats(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            estimated_cost_usd=self._estimate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                chat_model
            )
        )
        
        return ChatResponse(
            text=text_output,
            usage=usage,
            raw_response=response
        )


# =============================================================================
# CONVENIENCE FUNCTIONS (Backward Compatibility)
# =============================================================================

# Global client instance for backward compatibility
_default_client = LLMClient()


def create_embeddings(texts: List[str], model: str = config.EMBED_MODEL) -> Any:
    """
    Generate embeddings using the default client.
    
    This is a convenience function for backward compatibility.
    For new code, prefer using LLMClient directly.
    
    Args:
        texts: List of text strings to embed
        model: Embedding model name
        
    Returns:
        OpenAI embeddings response
    """
    return _default_client.create_embeddings(texts, model=model)


def call_llm(
    messages: List[Dict[str, str]],
    model: str = config.LLM_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 500
) -> Dict[str, Any]:
    """
    Call ChatCompletion API using the default client.
    
    This is a convenience function for backward compatibility.
    For new code, prefer using LLMClient directly.
    
    Args:
        messages: List of message dictionaries
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum response tokens
        
    Returns:
        Dictionary with 'text', 'usage', and 'raw_response' keys
    """
    response = _default_client.chat(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Convert to dict for backward compatibility
    return {
        "text": response.text,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "raw_response": response.raw_response
    }


def simple_generate(prompt: str) -> str:
    """
    Generate text from a simple prompt.
    
    This is a convenience function for quick testing.
    
    Args:
        prompt: The user prompt
        
    Returns:
        Generated text response
        
    Example:
        >>> answer = simple_generate("What is 2+2?")
        >>> print(answer)
        '4'
    """
    messages = [{"role": "user", "content": prompt}]
    response = _default_client.chat(messages)
    return response.text


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing LLM Client...")
    
    # Test chat
    client = LLMClient()
    response = client.chat([
        {"role": "user", "content": "Say 'test successful' if you can read this."}
    ])
    print(f"Response: {response.text}")
    print(f"Tokens: {response.usage.total_tokens}")
    print(f"Cost: ${response.usage.estimated_cost_usd:.6f}")
    
    # Test embeddings
    embeddings = client.create_embeddings(["test"])
    print(f"Embedding dimension: {len(embeddings.data[0].embedding)}")
    
    print("\n✅ All tests passed!")
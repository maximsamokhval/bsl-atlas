"""Factory for creating LLM provider instances."""

from typing import Any, Optional

from .base import LLMProvider
from .deepseek import DeepSeekProvider, LLMError


def create_llm_provider(
    provider: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """Create an LLM provider instance.

    Args:
        provider: Provider identifier ("deepseek").
        api_key: API key (required for DeepSeek).
        base_url: Custom base URL (optional).
        model: Model name override (optional).
        **kwargs: Additional provider-specific parameters.

    Returns:
        LLMProvider instance.

    Raises:
        ValueError: If provider is unknown or missing required parameters.
        LLMError: If provider initialization fails.
    """
    if provider == "deepseek":
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek provider")
        return DeepSeekProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=kwargs.get("timeout", 30.0),
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature", 0.7),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


__all__ = ["create_llm_provider", "LLMError"]

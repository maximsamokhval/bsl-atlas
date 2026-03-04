"""LLM provider protocol."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers.

    Any concrete LLM provider must implement at least the `generate` method.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt.

        Args:
            prompt: The input text prompt.
            **kwargs: Additional parameters (model, temperature, max_tokens, etc.)

        Returns:
            Generated text as a string.

        Raises:
            LLMError: If generation fails (API error, network, etc.)
        """
        ...

    async def async_generate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronous version of generate.

        If a provider does not support async, it may raise NotImplementedError.
        """
        ...

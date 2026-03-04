"""DeepSeek LLM provider implementation."""

from typing import Any, Optional

import httpx

from .base import LLMProvider


class LLMError(Exception):
    """Base exception for LLM provider errors."""

    pass


class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider."""

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ):
        """Initialize DeepSeek provider.

        Args:
            api_key: DeepSeek API key.
            base_url: Base URL for API (defaults to official DeepSeek API).
            model: Model identifier (defaults to "deepseek-chat").
            timeout: Request timeout in seconds.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0..2).
        """
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = httpx.Client(timeout=timeout)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt.

        Args:
            prompt: The input text prompt.
            **kwargs: Additional parameters (model, temperature, max_tokens, etc.)

        Returns:
            Generated text.

        Raises:
            LLMError: If generation fails.
        """
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        stream = kwargs.get("stream", False)

        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            if "choices" not in data or len(data["choices"]) == 0:
                raise LLMError("Invalid response format: missing choices")

            message = data["choices"][0].get("message", {})
            content = message.get("content", "").strip()
            if not content:
                raise LLMError("Empty response content")

            return content

        except httpx.HTTPStatusError as e:
            raise LLMError(f"HTTP error {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise LLMError(f"Network error: {e}") from e
        except (KeyError, ValueError, TypeError) as e:
            raise LLMError(f"Invalid response format: {e}") from e

    async def async_generate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronous version of generate.

        Note: This implementation uses httpx.AsyncClient for async requests.
        """
        import httpx as async_httpx

        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        stream = kwargs.get("stream", False)

        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with async_httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                if "choices" not in data or len(data["choices"]) == 0:
                    raise LLMError("Invalid response format: missing choices")

                message = data["choices"][0].get("message", {})
                content = message.get("content", "").strip()
                if not content:
                    raise LLMError("Empty response content")

                return content

            except async_httpx.HTTPStatusError as e:
                raise LLMError(f"HTTP error {e.response.status_code}: {e.response.text}") from e
            except async_httpx.RequestError as e:
                raise LLMError(f"Network error: {e}") from e
            except (KeyError, ValueError, TypeError) as e:
                raise LLMError(f"Invalid response format: {e}") from e

    def __repr__(self) -> str:
        return f"DeepSeekProvider(model={self.model}, base_url={self.base_url})"

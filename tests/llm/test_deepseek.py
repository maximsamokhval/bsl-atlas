"""Tests for DeepSeekProvider."""

from unittest.mock import Mock, patch

import pytest

from src.llm.deepseek import DeepSeekProvider


class TestDeepSeekProvider:
    """Test DeepSeek LLM provider."""

    def test_provider_implements_llmprovider_protocol(self):
        """DeepSeekProvider should implement LLMProvider protocol."""
        # This test will fail until DeepSeekProvider is implemented
        from src.llm.base import LLMProvider

        provider = DeepSeekProvider(api_key="test")
        assert isinstance(provider, LLMProvider)

    @patch("src.llm.deepseek.httpx.Client")
    def test_generate_sends_correct_request(self, mock_client_class):
        """generate should send proper HTTP request to DeepSeek API."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated response"}}]
        }
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = DeepSeekProvider(api_key="test-key")
        result = provider.generate("Hello, world!")

        # Verify request parameters
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "https://api.deepseek.com/v1/chat/completions" in call_args[0][0]
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"
        assert call_args[1]["json"]["model"] == "deepseek-chat"
        assert call_args[1]["json"]["messages"][0]["content"] == "Hello, world!"

        # Verify result
        assert result == "Generated response"

    @patch("src.llm.deepseek.httpx.Client")
    def test_generate_handles_api_error(self, mock_client_class):
        """generate should raise LLMError on API failure."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Exception("Unauthorized")
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = DeepSeekProvider(api_key="invalid-key")
        with pytest.raises(Exception, match="Unauthorized"):
            provider.generate("test")

    def test_initialization_with_custom_base_url(self):
        """DeepSeekProvider should accept custom base_url."""
        provider = DeepSeekProvider(api_key="key", base_url="https://custom.deepseek.com")
        assert provider.base_url == "https://custom.deepseek.com"

    def test_initialization_with_model_override(self):
        """DeepSeekProvider should accept model parameter."""
        provider = DeepSeekProvider(api_key="key", model="deepseek-coder")
        assert provider.model == "deepseek-coder"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

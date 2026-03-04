"""Tests for Deepinfra embedding provider."""

from unittest.mock import Mock, patch

import pytest

from src.indexer.embeddings import DeepinfraEmbeddings, create_embedding_provider


class TestDeepinfraEmbeddings:
    """Test Deepinfra embedding provider."""

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_initialization_without_api_key(self, mock_lc_openai):
        """DeepinfraEmbeddings can be initialized without an API key."""
        mock_instance = Mock()
        mock_lc_openai.return_value = mock_instance
        provider = DeepinfraEmbeddings(api_key=None)
        assert provider.model == "Qwen/Qwen3-Embedding-4B"
        assert provider.base_url == "https://api.deepinfra.com/v1/inference"
        mock_lc_openai.assert_called_once_with(
            api_key="no-key-required",
            model="Qwen/Qwen3-Embedding-4B",
            base_url="https://api.deepinfra.com/v1/inference",
        )

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_initialization_with_custom_model_and_url(self, mock_lc_openai):
        """DeepinfraEmbeddings accepts custom model and base_url."""
        mock_instance = Mock()
        mock_lc_openai.return_value = mock_instance
        provider = DeepinfraEmbeddings(
            api_key="test-key",
            model="custom/model",
            base_url="https://custom.deepinfra.com",
        )
        assert provider.model == "custom/model"
        assert provider.base_url == "https://custom.deepinfra.com"
        mock_lc_openai.assert_called_once_with(
            api_key="test-key",
            model="custom/model",
            base_url="https://custom.deepinfra.com",
        )

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_embed_documents(self, mock_lc_openai):
        """embed_documents delegates to underlying LangChain embeddings."""
        mock_instance = Mock()
        mock_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_lc_openai.return_value = mock_instance
        provider = DeepinfraEmbeddings()
        texts = ["hello", "world"]
        result = provider.embed_documents(texts)
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_instance.embed_documents.assert_called_once_with(texts)

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_embed_query(self, mock_lc_openai):
        """embed_query delegates to underlying LangChain embeddings."""
        mock_instance = Mock()
        mock_instance.embed_query.return_value = [0.5, 0.6]
        mock_lc_openai.return_value = mock_instance
        provider = DeepinfraEmbeddings()
        result = provider.embed_query("query")
        assert result == [0.5, 0.6]
        mock_instance.embed_query.assert_called_once_with("query")


class TestCreateEmbeddingProviderDeepinfra:
    """Test factory function for deepinfra provider."""

    def test_create_deepinfra_provider_without_key(self):
        """create_embedding_provider should accept deepinfra without API key."""
        with patch("src.indexer.embeddings.DeepinfraEmbeddings") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            provider = create_embedding_provider("deepinfra", api_key=None)
            assert provider is mock_instance
            mock_class.assert_called_once_with(
                api_key=None,
                model='Qwen/Qwen3-Embedding-4B',
                base_url=None,
            )

    def test_create_deepinfra_provider_with_key(self):
        """create_embedding_provider passes API key."""
        with patch("src.indexer.embeddings.DeepinfraEmbeddings") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            create_embedding_provider("deepinfra", api_key="test-key")
            mock_class.assert_called_once_with(
                api_key="test-key",
                model='Qwen/Qwen3-Embedding-4B',
                base_url=None,
            )

    def test_create_deepinfra_provider_with_custom_model(self):
        """create_embedding_provider respects model parameter."""
        with patch("src.indexer.embeddings.DeepinfraEmbeddings") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            create_embedding_provider("deepinfra", model="custom/model", api_key="key")
            mock_class.assert_called_once_with(
                api_key="key",
                model="custom/model",
                base_url=None,
            )

    def test_create_deepinfra_provider_with_custom_base_url(self):
        """create_embedding_provider respects base_url parameter."""
        with patch("src.indexer.embeddings.DeepinfraEmbeddings") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            create_embedding_provider("deepinfra", base_url="https://custom.url", api_key="key")
            mock_class.assert_called_once_with(
                api_key="key",
                model='Qwen/Qwen3-Embedding-4B',
                base_url="https://custom.url",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

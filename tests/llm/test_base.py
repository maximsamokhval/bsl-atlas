"""Tests for LLMProvider protocol."""


from src.llm.base import LLMProvider


class TestLLMProvider:
    """Test that LLMProvider protocol is correctly defined."""

    def test_protocol_has_generate_method(self):
        """LLMProvider must have a generate method."""
        # This test will fail if LLMProvider doesn't have generate
        assert hasattr(LLMProvider, "generate")
        # Check signature (optional)
        # import inspect
        # sig = inspect.signature(LLMProvider.generate)
        # assert "prompt" in sig.parameters

    def test_protocol_is_runtime_checkable(self):
        """LLMProvider should be runtime checkable for isinstance checks."""
        # Use a concrete provider that implements the protocol
        from src.llm.deepseek import DeepSeekProvider

        provider = DeepSeekProvider(api_key="test")
        # If protocol is runtime_checkable, isinstance should work
        assert isinstance(provider, LLMProvider)

    def test_concrete_class_implements_protocol(self):
        """A dummy class that does NOT implement the protocol should fail."""

        class BadProvider:
            pass

        # With runtime_checkable, isinstance should return False, not raise TypeError
        assert not isinstance(BadProvider(), LLMProvider)

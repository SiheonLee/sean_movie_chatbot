from __future__ import annotations

import unittest
from unittest.mock import patch

from rag.config import settings
from rag.providers import get_llm


class AnthropicProviderTests(unittest.TestCase):
    def test_anthropic_requires_api_key(self):
        with (
            patch.object(settings, "llm_provider", "anthropic"),
            patch.object(settings, "anthropic_api_key", None),
        ):
            with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
                get_llm()

    def test_anthropic_uses_configured_haiku_model(self):
        with (
            patch.object(settings, "llm_provider", "anthropic"),
            patch.object(settings, "anthropic_api_key", "test-key"),
            patch.object(settings, "anthropic_model", "claude-haiku-4-5-20251001"),
            patch.object(settings, "llm_temperature", 0.1),
            patch.object(settings, "llm_max_tokens", 512),
            patch("langchain_anthropic.ChatAnthropic") as chat_anthropic,
        ):
            get_llm()

        chat_anthropic.assert_called_once_with(
            model="claude-haiku-4-5-20251001",
            api_key="test-key",
            temperature=0.1,
            max_tokens=512,
        )


if __name__ == "__main__":
    unittest.main()

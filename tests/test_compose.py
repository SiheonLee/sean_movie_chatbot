from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


class ComposeDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compose = (ROOT_DIR / "compose.yaml").read_text(encoding="utf-8")
        cls.api_block, cls.ui_block = cls.compose.split("\n  ui:", maxsplit=1)

    def test_api_is_only_available_on_the_compose_network(self):
        self.assertNotIn("\n    ports:", self.api_block)
        self.assertIn("RAG_API_URL: http://api:8000", self.ui_block)

    def test_ui_requires_and_receives_the_shared_passcode(self):
        self.assertIn(
            "CINEBOT_PASSCODE: ${CINEBOT_PASSCODE:?set CINEBOT_PASSCODE in .env}",
            self.ui_block,
        )

    def test_persistent_request_limit_db_uses_the_existing_state_volume(self):
        self.assertIn("RATE_LIMIT_DB: /app/state/rate_limits.sqlite", self.api_block)
        self.assertIn("checkpoint_data:/app/state", self.api_block)

    def test_limit_overrides_are_shared_by_api_and_ui(self):
        self.assertEqual(
            self.compose.count('DAILY_QUESTION_LIMIT: "${DAILY_QUESTION_LIMIT:-30}"'),
            2,
        )
        self.assertEqual(
            self.compose.count(
                'SESSION_QUESTION_LIMIT: "${SESSION_QUESTION_LIMIT:-12}"'
            ),
            2,
        )
        self.assertIn(
            'MAX_CONCURRENT_REQUESTS: "${MAX_CONCURRENT_REQUESTS:-2}"',
            self.api_block,
        )


if __name__ == "__main__":
    unittest.main()

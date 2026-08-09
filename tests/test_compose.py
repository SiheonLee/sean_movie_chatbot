from __future__ import annotations

import unittest
from pathlib import Path

import yaml


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

    def test_chroma_seed_stays_read_only_and_runtime_writes_use_tmpfs(self):
        self.assertIn("target: /app/chroma_seed", self.api_block)
        self.assertIn("read_only: true", self.api_block)
        self.assertIn("CHROMA_DIR: /app/chroma_runtime", self.api_block)
        self.assertIn("cp -a /app/chroma_seed/. /app/chroma_runtime/", self.api_block)
        self.assertIn("/app/chroma_runtime:size=64m,mode=0755", self.api_block)

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


class Ec2ComposeDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compose_path = ROOT_DIR / "compose.ec2.yaml"
        cls.compose = yaml.safe_load(cls.compose_path.read_text(encoding="utf-8"))
        cls.services = cls.compose["services"]
        cls.caddyfile = (ROOT_DIR / "Caddyfile").read_text(encoding="utf-8")

    def test_only_caddy_publishes_host_ports(self):
        self.assertNotIn("ports", self.services["api"])
        self.assertNotIn("ports", self.services["ui"])
        self.assertEqual(self.services["caddy"]["ports"], ["80:80", "443:443"])

    def test_caddy_only_reaches_the_ui_network(self):
        self.assertEqual(self.services["api"]["networks"], ["backend"])
        self.assertEqual(
            set(self.services["ui"]["networks"]), {"frontend", "backend"}
        )
        self.assertEqual(self.services["caddy"]["networks"], ["frontend"])

    def test_caddy_routes_the_public_domain_to_streamlit(self):
        self.assertIn("seandev27-cinebot.duckdns.org", self.caddyfile)
        self.assertIn("reverse_proxy ui:8501", self.caddyfile)
        self.assertNotIn("api:8000", self.caddyfile)

    def test_search_artifacts_are_read_only_bind_mounts(self):
        mounts = {
            mount["target"]: mount
            for mount in self.services["api"]["volumes"]
            if isinstance(mount, dict)
        }
        for target in (
            "/app/catalog/movies.json",
            "/app/catalog/enriched.json",
            "/app/chroma_seed",
        ):
            with self.subTest(target=target):
                self.assertEqual(mounts[target]["type"], "bind")
                self.assertTrue(mounts[target]["read_only"])
                self.assertFalse(mounts[target]["bind"]["create_host_path"])

    def test_chroma_runtime_copy_is_ephemeral_and_writable(self):
        api = self.services["api"]
        self.assertEqual(api["environment"]["CHROMA_DIR"], "/app/chroma_runtime")
        self.assertIn(
            "cp -a /app/chroma_seed/. /app/chroma_runtime/",
            api["command"][2],
        )
        self.assertEqual(api["tmpfs"], ["/app/chroma_runtime:size=64m,mode=0755"])

    def test_runtime_and_certificate_data_use_named_volumes(self):
        api_volumes = self.services["api"]["volumes"]
        ui_volumes = self.services["ui"]["volumes"]
        caddy_volumes = self.services["caddy"]["volumes"]
        self.assertIn("checkpoint_data:/app/state", api_volumes)
        self.assertIn("user_data:/app/user_data", ui_volumes)
        self.assertIn("caddy_data:/data", caddy_volumes)
        self.assertIn("caddy_config:/config", caddy_volumes)
        self.assertEqual(
            set(self.compose["volumes"]),
            {"checkpoint_data", "user_data", "caddy_data", "caddy_config"},
        )

    def test_shared_passcode_and_low_request_limits_remain_enabled(self):
        ui_environment = self.services["ui"]["environment"]
        api_environment = self.services["api"]["environment"]
        self.assertIn("set CINEBOT_PASSCODE in .env", ui_environment["CINEBOT_PASSCODE"])
        self.assertEqual(api_environment["DAILY_QUESTION_LIMIT"], "${DAILY_QUESTION_LIMIT:-30}")
        self.assertEqual(api_environment["SESSION_QUESTION_LIMIT"], "${SESSION_QUESTION_LIMIT:-12}")
        self.assertEqual(api_environment["MAX_CONCURRENT_REQUESTS"], "${MAX_CONCURRENT_REQUESTS:-2}")

    def test_app_image_is_pinned_to_the_d6_build(self):
        expected = "${CINEBOT_IMAGE:-seandev27/cinebot:d6-cbd7a10}"
        self.assertEqual(self.services["api"]["image"], expected)
        self.assertEqual(self.services["ui"]["image"], expected)
        self.assertEqual(self.services["caddy"]["image"], "caddy:2.11.4-alpine")


if __name__ == "__main__":
    unittest.main()

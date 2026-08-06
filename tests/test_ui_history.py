from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ui import history


def messages(*questions: str) -> list[dict]:
    turns: list[dict] = []
    for question in questions:
        turns.append({"role": "user", "content": question})
        turns.append(
            {
                "role": "assistant",
                "content": f"{question}에 대한 답변",
                "sources": [{"title": "기생충", "year": 2019}],
            }
        )
    return turns


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = patch.object(history, "HISTORY_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_saved_conversation_comes_back(self):
        history.save_conversation("abc-1", messages("기생충 감독은?"))

        loaded = history.load_conversation("abc-1")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded[0]["content"], "기생충 감독은?")
        # 출처 카드까지 그대로 살아야 화면을 다시 그릴 수 있다.
        self.assertEqual(loaded[1]["sources"][0]["title"], "기생충")

    def test_title_is_the_first_question(self):
        history.save_conversation("abc-2", messages("한국 스릴러 추천해줘", "다른 것도"))

        summary = history.list_conversations()[0]

        self.assertEqual(summary.title, "한국 스릴러 추천해줘")
        self.assertEqual(summary.turns, 2)

    def test_newest_first(self):
        history.save_conversation("old", messages("옛날 질문"))
        history.save_conversation("new", messages("최근 질문"))
        # updated_at은 초 단위라 같은 값이 될 수 있다. 명시적으로 벌린다.
        path = history.HISTORY_DIR / "old.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["updated_at"] = "2020-01-01T00:00:00"
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

        titles = [s.title for s in history.list_conversations()]

        self.assertEqual(titles, ["최근 질문", "옛날 질문"])

    def test_empty_conversation_is_not_saved(self):
        history.save_conversation("empty", [])

        self.assertEqual(history.list_conversations(), [])

    def test_path_traversal_is_rejected(self):
        """conversation_id가 파일 이름이 된다. 검사 없이 붙이면 밖으로 나갈 수 있다."""
        history.save_conversation("../escape", messages("질문"))

        self.assertEqual(history.list_conversations(), [])
        self.assertIsNone(history.load_conversation("../escape"))

    def test_broken_file_does_not_break_the_list(self):
        history.save_conversation("good", messages("멀쩡한 질문"))
        (history.HISTORY_DIR / "broken.json").write_text("{못 읽는", encoding="utf-8")

        titles = [s.title for s in history.list_conversations()]

        self.assertEqual(titles, ["멀쩡한 질문"])

    def test_old_schema_is_ignored(self):
        history.save_conversation("v0", messages("질문"))
        path = history.HISTORY_DIR / "v0.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["version"] = 0
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

        self.assertEqual(history.list_conversations(), [])
        self.assertIsNone(history.load_conversation("v0"))

    def test_other_users_records_are_invisible(self):
        """멀티 유저가 되면 이 경계가 남의 대화를 막는다."""
        history.save_conversation("theirs", messages("남의 질문"), user_id="someone")

        self.assertEqual(history.list_conversations(), [])
        self.assertIsNone(history.load_conversation("theirs"))
        self.assertIsNotNone(history.load_conversation("theirs", user_id="someone"))

    def test_old_conversations_are_pruned(self):
        with patch.object(history, "MAX_CONVERSATIONS", 3):
            for index in range(5):
                history.save_conversation(f"c{index}", messages(f"질문{index}"))
                path = history.HISTORY_DIR / f"c{index}.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                record["updated_at"] = f"2026-01-0{index + 1}T00:00:00"
                path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            history.save_conversation("c9", messages("마지막"))

        remaining = {s.conversation_id for s in history.list_conversations()}
        self.assertEqual(len(remaining), 3)
        self.assertIn("c9", remaining)

    def test_no_partial_file_is_left_behind(self):
        history.save_conversation("atomic", messages("질문"))

        leftovers = list(history.HISTORY_DIR.glob("*.tmp"))
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()

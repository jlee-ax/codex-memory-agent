from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from codex_memory_agent import AgentStateError, TaxonomyAgent
from codex_memory_agent.transport import JSON, CodexConnection, Message, TurnCancelled


def mapping(value: JSON) -> Message:
    assert isinstance(value, dict)
    return value


def rows(value: JSON) -> list[Message]:
    assert isinstance(value, list)
    return [mapping(row) for row in value]


class MemoryDrivenConnection(CodexConnection):
    """In-process model substitute exercising the real SQLite tool contract."""

    def __init__(self, mode: str = "complete") -> None:
        self.mode = mode
        self.calls = 0
        self.was_closed = False
        self.entered = threading.Event()
        self.revised = False
        self.unread_attempts: list[JSON] = []

    def run_turn(
        self,
        prompt: str,
        dynamic_tools: list[Message],
        tool_handler: Callable[[str, Message], JSON],
        model: str | None = None,
        timeout: float = 300,
        cancel_event: threading.Event | None = None,
        on_event: Callable[[Message], None] | None = None,
    ) -> str:
        self.calls += 1
        status = mapping(tool_handler("memory_status", {}))
        if self.mode == "stall":
            return "No decisions."
        if self.mode == "block":
            self.entered.set()
            assert cancel_event is not None
            assert cancel_event.wait(timeout=5)
            raise TurnCancelled("Stopped")
        if status["active_categories"] == 0:
            tool_handler(
                "edit_categories",
                {
                    "expected_revision": status["revision"],
                    "actions": [
                        {
                            "action": "create",
                            "category_ids": [],
                            "expected_versions": [],
                            "categories": [
                                {"name": "Writing", "definition": "Produce written material."}
                            ],
                            "evidence_ids": [1],
                            "reason": "The first original is a writing task.",
                        }
                    ],
                },
            )
            status = mapping(tool_handler("memory_status", {}))
        category = rows(mapping(tool_handler("read_categories", {}))["rows"])[0]
        if status["phase"] == "review" and self.mode == "revise" and not self.revised:
            tool_handler("review_control", {"action": "reopen"})
            tool_handler(
                "edit_categories",
                {
                    "expected_revision": status["revision"],
                    "actions": [
                        {
                            "action": "revise",
                            "category_ids": [category["category_id"]],
                            "expected_versions": [category["version"]],
                            "categories": [
                                {
                                    "name": "Writing",
                                    "definition": "Create or edit written material.",
                                }
                            ],
                            "evidence_ids": [1],
                            "reason": "Refine the boundary after review.",
                        }
                    ],
                },
            )
            self.revised = True
            status = mapping(tool_handler("memory_status", {}))
            category = rows(mapping(tool_handler("read_categories", {}))["rows"])[0]
        if self.mode == "unread":
            self.unread_attempts.append(
                tool_handler(
                    "assign_items",
                    {
                        "expected_revision": status["revision"],
                        "assignments": [
                            {
                                "item_id": 1,
                                "category_id": category["category_id"],
                                "category_version": category["version"],
                                "reason": "Unread attempt.",
                            }
                        ],
                    },
                )
            )
        queue = "review" if status["phase"] == "review" else "work"
        items = rows(mapping(tool_handler("next_items", {"queue": queue, "limit": 200}))["rows"])
        assignments: list[JSON] = [
            {
                "item_id": item["item_id"],
                "category_id": category["category_id"],
                "category_version": category["version"],
                "reason": "The original requires writing.",
            }
            for item in items
        ]
        response = mapping(
            tool_handler(
                "assign_items",
                {
                    "expected_revision": status["revision"],
                    "assignments": assignments,
                },
            )
        )
        assert "error" not in response
        return "Decisions saved."

    def close(self) -> None:
        self.was_closed = True


def test_run_returns_results_after_all_item_final_review(tmp_path: Path) -> None:
    # Given
    connection = MemoryDrivenConnection()
    with TaxonomyAgent(connection) as agent:
        # When
        result = agent.run(["Write reports", "Write manuals"], "Group writing", directory=tmp_path)
        # Then
        assert mapping(result["status"])["complete"] is True
        assert mapping(result["status"])["reviewed"] == 2
        assert connection.calls == 2
        assert {row["raw_text"] for row in rows(result["assignments"])} == {
            "Write reports",
            "Write manuals",
        }
        assert result["k"] == 1


def test_same_raw_strings_and_prompt_resume_without_model_calls(tmp_path: Path) -> None:
    # Given
    connection = MemoryDrivenConnection()
    with TaxonomyAgent(connection) as agent:
        agent.run([" Write reports ", "Write manuals"], "Group writing", directory=tmp_path)
        previous_database = agent.database
        # When
        result = agent.run(
            ["Write manuals", " Write reports ", "Write manuals"],
            "Group writing",
            directory=tmp_path,
        )
        # Then
        assert agent.database == previous_database
        assert connection.calls == 2
        assert mapping(result["status"])["complete"] is True
        assert {row["raw_text"] for row in rows(result["assignments"])} == {
            " Write reports ",
            "Write manuals",
        }


@pytest.mark.parametrize(
    "texts,prompt", [(["Write reports"], "By output"), (["write reports"], "By purpose")]
)
def test_changed_input_gets_separate_empty_taxonomy(
    tmp_path: Path, texts: list[str], prompt: str
) -> None:
    # Given
    with TaxonomyAgent(MemoryDrivenConnection()) as agent:
        agent.prepare(["Write reports"], "By purpose", directory=tmp_path)
        previous_database = agent.database
        # When
        state = agent.prepare(texts, prompt, directory=tmp_path)
        # Then
        assert agent.database != previous_database
        assert state["pending"] == 1
        assert state["active_categories"] == 0
        assert state["k"] == 0


def test_wait_propagates_stall_without_claiming_completion(tmp_path: Path) -> None:
    # Given
    with TaxonomyAgent(MemoryDrivenConnection("stall")) as agent:
        agent.start(["Write reports"], "Group work", directory=tmp_path)
        # When
        with pytest.raises(AgentStateError, match="no persisted progress"):
            agent.wait(timeout=5)
        # Then
        assert agent.status()["complete"] is False
        assert agent.status()["pending"] == 1
        assert agent.status()["turns"] == 3
        assert str(agent.status()["error"]).startswith("AgentStateError:")


def test_each_turn_must_fetch_originals_before_assigning(tmp_path: Path) -> None:
    # Given
    connection = MemoryDrivenConnection("unread")
    with TaxonomyAgent(connection) as agent:
        # When
        result = agent.run(["Write reports"], "Group work", directory=tmp_path)
        # Then
        assert len(connection.unread_attempts) == 2
        assert all("error" in mapping(response) for response in connection.unread_attempts)
        assert mapping(result["status"])["complete"] is True


def test_category_search_does_not_mark_items_as_read(tmp_path: Path) -> None:
    # Given
    with TaxonomyAgent(MemoryDrivenConnection()) as agent:
        agent.prepare(["Write reports"], "Group work", directory=tmp_path)
        agent._handle_tool(
            "edit_categories",
            {
                "expected_revision": 0,
                "actions": [
                    {
                        "action": "create",
                        "category_ids": [],
                        "expected_versions": [],
                        "categories": [
                            {"name": "Writing", "definition": "Produce written material."}
                        ],
                        "evidence_ids": [1],
                        "reason": "Example evidence.",
                    }
                ],
            },
        )
        # When
        response = agent._handle_tool(
            "search_memory", {"target": "categories", "query": "Writing", "mode": "literal"}
        )
        # Then
        assert len(rows(response["rows"])) == 1
        assert agent._seen_items == set()


def test_item_search_allows_read_before_assign(tmp_path: Path) -> None:
    # Given
    with TaxonomyAgent(MemoryDrivenConnection()) as agent:
        agent.prepare(["Write reports"], "Group work", directory=tmp_path)
        # When
        response = agent._handle_tool(
            "search_memory", {"target": "items", "query": "reports", "mode": "literal"}
        )
        # Then
        assert len(rows(response["rows"])) == 1
        assert agent._seen_items == {1}


def test_revision_restarts_review_against_new_category_version(tmp_path: Path) -> None:
    # Given
    connection = MemoryDrivenConnection("revise")
    with TaxonomyAgent(connection) as agent:
        # When
        result = agent.run(["Write reports", "Write manuals"], "Group work", directory=tmp_path)
        # Then
        assert connection.calls == 3
        assert mapping(result["status"])["complete"] is True
        assert mapping(result["status"])["reviewed"] == 2
        assert {row["category_version"] for row in rows(result["assignments"])} == {2}


def test_stop_preserves_incomplete_results(tmp_path: Path) -> None:
    # Given
    connection = MemoryDrivenConnection("block")
    with TaxonomyAgent(connection) as agent:
        agent.start(["Write reports"], "Group work", directory=tmp_path)
        assert connection.entered.wait(timeout=5)
        # When
        stopped = agent.stop(timeout=5)
        # Then
        assert stopped["running"] is False
        assert stopped["complete"] is False
        assert mapping(agent.wait()["status"])["pending"] == 1


def test_wait_timeout_does_not_cancel_active_job(tmp_path: Path) -> None:
    # Given
    connection = MemoryDrivenConnection("block")
    with TaxonomyAgent(connection) as agent:
        agent.start(["Write reports"], "Group work", directory=tmp_path)
        assert connection.entered.wait(timeout=5)
        # When
        with pytest.raises(TimeoutError):
            agent.wait(timeout=0)
        # Then
        assert agent.running
        assert not agent._cancel.is_set()


@pytest.mark.parametrize("own_connection", [False, True])
def test_context_manager_respects_connection_ownership(own_connection: bool) -> None:
    # Given
    connection = MemoryDrivenConnection()
    # When
    with TaxonomyAgent(connection, own_connection=own_connection) as agent:
        assert not agent.closed
    # Then
    assert agent.closed
    assert connection.was_closed is own_connection


def test_caller_supplied_connection_remains_open_by_default() -> None:
    # Given
    connection = MemoryDrivenConnection()
    # When
    with TaxonomyAgent(connection):
        pass
    # Then
    assert not connection.was_closed


def test_constructed_connection_is_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    connection = MemoryDrivenConnection()
    monkeypatch.setattr("codex_memory_agent.agent.CodexConnection", lambda: connection)
    # When
    with TaxonomyAgent():
        pass
    # Then
    assert connection.was_closed


def test_wait_requires_started_job() -> None:
    # Given
    with TaxonomyAgent(MemoryDrivenConnection()) as agent:
        # When / Then
        with pytest.raises(AgentStateError, match="Start a job"):
            agent.wait()


def test_monitor_reports_optional_dependency_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    with TaxonomyAgent(MemoryDrivenConnection()) as agent:
        # When / Then
        with pytest.raises(AgentStateError, match=r"codex-memory-agent\[notebook\]"):
            agent.monitor()

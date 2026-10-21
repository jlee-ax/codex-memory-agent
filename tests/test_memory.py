"""Real temporary SQLite scenarios exercising the public memory tools."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from codex_memory_agent.memory import MemoryConflict, Record, TaxonomyMemory


@pytest.fixture
def memory(tmp_path: Path) -> Iterator[TaxonomyMemory]:
    with TaxonomyMemory(tmp_path / "memory.sqlite") as store:
        store.ingest([" raw ", "raw", "다른 원문"])
        yield store


def edit(
    store: TaxonomyMemory,
    action: str = "create",
    ids: tuple[int, ...] = (),
    versions: tuple[int, ...] = (),
    names: tuple[str, ...] = ("A",),
) -> Record:
    return store.call_tool(
        "edit_categories",
        {
            "expected_revision": store.snapshot()["revision"],
            "actions": [
                {
                    "action": action,
                    "category_ids": list(ids),
                    "expected_versions": list(versions),
                    "categories": [{"name": n, "definition": n + " definition"} for n in names],
                    "evidence_ids": [1],
                    "reason": "Observed raw item evidence",
                }
            ],
        },
    )


def assign(
    store: TaxonomyMemory,
    items: tuple[int, ...] = (1, 2, 3),
    category_id: int = 1,
    version: int = 1,
) -> Record:
    return store.call_tool(
        "assign_items",
        {
            "expected_revision": store.snapshot()["revision"],
            "assignments": [
                {
                    "item_id": item,
                    "category_id": category_id,
                    "category_version": version,
                    "reason": "Definition matches this raw text",
                }
                for item in items
            ],
        },
    )


def test_raw_text_and_dataset_fingerprint_survive_resume(tmp_path: Path) -> None:
    # Given immutable strings with meaningful spaces and duplicate source rows.
    path = tmp_path / "resume.sqlite"
    with TaxonomyMemory(path) as store:
        store.ingest([" raw ", "raw", "raw"])
    # When the identical unique dataset is resumed in a different order.
    with TaxonomyMemory(path) as store:
        store.ingest(["raw", " raw "])
        rows = store.call_tool("next_items", {"queue": "all"})["rows"]
    # Then stable IDs preserve both distinct raw strings.
    assert [(r["item_id"], r["raw_text"]) for r in rows] == [(1, " raw "), (2, "raw")]


def test_resume_rejects_changed_dataset(memory: TaxonomyMemory) -> None:
    # Given a populated memory.
    before = memory.exports()
    # When a different dataset attempts to resume.
    with pytest.raises(MemoryConflict, match="fingerprint"):
        memory.ingest(["changed"])
    # Then persistent content is unchanged.
    assert memory.exports() == before


def test_assignment_batch_rolls_back_on_unknown_item(memory: TaxonomyMemory) -> None:
    # Given an active category and unassigned items.
    edit(memory)
    before = memory.exports()
    # When a later row in one batch is invalid.
    with pytest.raises(MemoryConflict, match="Unknown item"):
        assign(memory, items=(1, 999))
    # Then neither the earlier assignment nor its history persists.
    assert memory.exports() == before


def test_edit_batch_rolls_back_versions_revision_and_history(
    memory: TaxonomyMemory,
) -> None:
    # Given an existing category.
    edit(memory)
    before, status = memory.exports(), memory.snapshot()
    action = {
        "action": "revise",
        "category_ids": [1],
        "expected_versions": [1],
        "categories": [{"name": "updated", "definition": "updated definition"}],
        "evidence_ids": [1],
        "reason": "revision evidence",
    }
    # When the second edit uses a now-obsolete version in the same transaction.
    with pytest.raises(MemoryConflict, match="version conflict"):
        memory.call_tool("edit_categories", {"expected_revision": 1, "actions": [action, action]})
    # Then no part of either edit remains.
    assert (memory.exports(), memory.snapshot()) == (before, status)


def test_revision_invalidates_audits_in_other_categories(
    memory: TaxonomyMemory,
) -> None:
    # Given completed audits spanning two categories.
    edit(memory)
    edit(memory, names=("B",))
    assign(memory, items=(1, 2))
    assign(memory, items=(3,), category_id=2)
    memory.call_tool("review_control", {"action": "begin"})
    assign(memory, items=(1, 2))
    assign(memory, items=(3,), category_id=2)
    # When only category A is revised.
    edit(memory, "revise", (1,), (1,), ("A revised",))
    # Then all audits expire but only A's memberships are stale.
    status = memory.snapshot()
    assert (status["needs_review"], status["stale"], status["stage"]) == (
        3,
        2,
        "building",
    )
    assert len(memory.exports()["category_versions"]) == 3


@pytest.mark.parametrize(
    ("operation", "sources", "names"),
    [("merge", (1, 2), ("combined",)), ("split", (1,), ("left", "right"))],
)
def test_merge_and_split_queue_affected_memberships(
    memory: TaxonomyMemory,
    operation: str,
    sources: tuple[int, ...],
    names: tuple[str, ...],
) -> None:
    # Given assigned source categories.
    edit(memory)
    edit(memory, names=("B",))
    assign(memory, items=(1, 2))
    assign(memory, items=(3,), category_id=2)
    # When category topology changes.
    edit(memory, operation, sources, tuple(1 for _ in sources), names)
    # Then retired-source memberships are explicitly returned for reconsideration.
    rows = memory.call_tool("next_items", {"queue": "work"})["rows"]
    expected = [1, 2, 3] if operation == "merge" else [1, 2]
    assert [row["item_id"] for row in rows] == expected
    assert all(row["stale"] == 1 for row in rows)


def test_finalize_refuses_incomplete_global_review(memory: TaxonomyMemory) -> None:
    # Given all assignments but only a partial final audit.
    edit(memory)
    assign(memory)
    memory.call_tool("review_control", {"action": "begin"})
    assign(memory, items=(1, 2))
    # When finalization is requested.
    with pytest.raises(MemoryConflict, match="Exhaustive"):
        memory.call_tool("review_control", {"action": "finalize"})
    # Then the remaining item stays visible for review.
    assert [r["item_id"] for r in memory.call_tool("next_items", {"queue": "review"})["rows"]] == [
        3
    ]


def test_finalize_freezes_assignments_until_reopen(memory: TaxonomyMemory) -> None:
    # Given an exhaustively audited and finalized taxonomy.
    edit(memory)
    assign(memory)
    memory.call_tool("review_control", {"action": "begin"})
    assign(memory)
    memory.call_tool("review_control", {"action": "finalize"})
    # When an assignment attempts to change a frozen result.
    with pytest.raises(MemoryConflict, match="frozen"):
        assign(memory, items=(1,))
    # Then completion remains persisted.
    assert memory.snapshot()["complete"] is True


def test_reopen_resets_audits(memory: TaxonomyMemory) -> None:
    # Given a finalized memory.
    edit(memory)
    assign(memory)
    memory.call_tool("review_control", {"action": "begin"})
    assign(memory)
    memory.call_tool("review_control", {"action": "finalize"})
    # When explicitly reopened.
    result = memory.call_tool("review_control", {"action": "reopen"})
    # Then a fresh exhaustive review is required.
    assert (result["complete"], result["needs_review"], result["phase"]) == (
        False,
        3,
        "discovery",
    )


@pytest.mark.parametrize("mode", ["literal", "fts"])
def test_search_pagination_reports_exhaustive_total(memory: TaxonomyMemory, mode: str) -> None:
    # Given two exact raw strings matching the term.
    # When retrieving one result at a time.
    first = memory.call_tool("search_memory", {"mode": mode, "query": "raw", "limit": 1})
    second = memory.call_tool(
        "search_memory", {"mode": mode, "query": "raw", "limit": 1, "offset": 1}
    )
    # Then total exposes the untruncated scope and pages are distinct.
    assert (first["total"], second["total"]) == (2, 2)
    assert first["rows"][0]["item_id"] != second["rows"][0]["item_id"]


def test_worker_thread_can_write_while_main_reads(memory: TaxonomyMemory) -> None:
    # Given memory constructed by the notebook's main thread.
    from concurrent.futures import ThreadPoolExecutor

    edit(memory)
    # When the background executor writes and the main thread polls progress.
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(assign, memory)
        snapshots = [memory.snapshot() for _ in range(10)]
        result = future.result()
    # Then cross-thread access succeeds and each read sees a complete transaction.
    assert result["pending"] == 0
    assert all(status["pending"] in (0, 3) for status in snapshots)


def test_category_evidence_available_in_current_and_version_exports(
    memory: TaxonomyMemory,
) -> None:
    # Given an evidence-backed category.
    edit(memory)
    # When its definition is revised.
    edit(memory, "revise", (1,), (1,), ("revised",))
    # Then both definitions retain their raw evidence IDs for replay.
    versions = memory.exports()["category_versions"]
    assert [(r["version"], r["evidence_ids"]) for r in versions] == [(1, [1]), (2, [1])]
    assert memory.call_tool("read_categories", {})["rows"][0]["evidence_ids"] == [1]


def test_review_refuses_unseen_items(memory: TaxonomyMemory) -> None:
    # Given all raw items remain unassigned.
    # When a final review is requested.
    with pytest.raises(MemoryConflict, match="Unassigned"):
        memory.call_tool("review_control", {"action": "begin"})
    # Then no item has disappeared from the work queue.
    assert memory.call_tool("next_items", {"queue": "work"})["total"] == 3


def test_obsolete_global_revision_rejects_assignment(memory: TaxonomyMemory) -> None:
    # Given taxonomy revision advanced after the caller observed zero.
    edit(memory)
    # When the old request arrives.
    with pytest.raises(MemoryConflict, match="Global revision"):
        memory.call_tool(
            "assign_items",
            {
                "expected_revision": 0,
                "assignments": [
                    {
                        "item_id": 1,
                        "category_id": 1,
                        "category_version": 1,
                        "reason": "old request",
                    }
                ],
            },
        )
    # Then the stale caller cannot modify membership.
    assert memory.snapshot()["pending"] == 3


def test_history_tool_paginates_prior_definitions(memory: TaxonomyMemory) -> None:
    # Given a category whose definition was revised.
    edit(memory)
    edit(memory, "revise", (1,), (1,), ("revised",))
    # When the model requests the retained first definition.
    page = memory.call_tool("read_categories", {"category_ids": [1], "history": True, "limit": 1})
    # Then the old definition and the total version count remain accessible.
    assert (page["total"], page["rows"][0]["version"], page["rows"][0]["name"]) == (
        2,
        1,
        "A",
    )


def test_k_counts_only_active_categories_with_members(memory: TaxonomyMemory) -> None:
    # Given two active categories but only one receives members.
    edit(memory)
    edit(memory, names=("empty",))
    # When items are assigned to the first category.
    result = assign(memory)
    # Then k reflects used categories while catalog size remains available.
    assert (result["k"], result["active_categories"]) == (1, 2)


def test_taxonomy_edit_preserves_audit_history_but_invalidates_review(
    memory: TaxonomyMemory,
) -> None:
    # Given all items audited at revision one.
    edit(memory)
    assign(memory)
    memory.call_tool("review_control", {"action": "begin"})
    assign(memory)
    # When a new category advances the global taxonomy revision.
    edit(memory, names=("new category",))
    # Then old audit stamps remain readable while every item requires review.
    rows = memory.call_tool("next_items", {"queue": "review"})["rows"]
    assert len(rows) == 3
    assert {row["audit_revision"] for row in rows} == {1}
    assert memory.snapshot()["revision"] == 2


def test_page_budget_preserves_all_raw_strings_across_offsets(tmp_path: Path) -> None:
    # Given rows whose combined JSON exceeds a page, including one oversized row.
    import json

    texts = ["a" * 70_000, "b" * 35_000, "c" * 35_000, "d" * 100]
    with TaxonomyMemory(tmp_path / "large.sqlite") as store:
        store.ingest(texts)
        pages = []
        offset = 0
        # When every page is fetched using the explicit continuation offset.
        while offset is not None:
            page = store.call_tool("next_items", {"queue": "all", "offset": offset})
            pages.append(page)
            offset = page["next_offset"]
    # Then no raw characters or items are lost and large singleton rows are explicit.
    assert [row["raw_text"] for page in pages for row in page["rows"]] == texts
    assert all(
        len(json.dumps(page, ensure_ascii=True)) <= 60_000
        or (page["oversized"] is True and len(page["rows"]) == 1)
        for page in pages
    )
    assert pages[0]["oversized"] is True
    assert len(pages) == 3


@pytest.mark.parametrize("mode", ["literal", "fts"])
def test_category_search_replaces_obsolete_definition(memory: TaxonomyMemory, mode: str) -> None:
    # Given a category with an old searchable definition.
    edit(memory, names=("stable",))
    # When only its definition is revised while its name stays the same.
    memory.call_tool(
        "edit_categories",
        {
            "expected_revision": 1,
            "actions": [
                {
                    "action": "revise",
                    "category_ids": [1],
                    "expected_versions": [1],
                    "categories": [{"name": "stable", "definition": "currentmeaning"}],
                    "evidence_ids": [1],
                    "reason": "Updated semantic boundary",
                }
            ],
        },
    )
    # Then current definition search succeeds and obsolete terms disappear.
    current = memory.call_tool(
        "search_memory",
        {"target": "categories", "mode": mode, "query": "currentmeaning"},
    )
    obsolete = memory.call_tool(
        "search_memory", {"target": "categories", "mode": mode, "query": "definition"}
    )
    assert (current["total"], obsolete["total"]) == (1, 0)
    assert current["rows"][0]["version"] == 2


@pytest.mark.parametrize("mode", ["literal", "fts"])
@pytest.mark.parametrize(
    ("operation", "sources", "names"),
    [
        ("merge", (1, 2), ("replacement",)),
        ("split", (1,), ("replacement1", "replacement2")),
    ],
)
def test_category_search_excludes_retired_categories(
    memory: TaxonomyMemory,
    mode: str,
    operation: str,
    sources: tuple[int, ...],
    names: tuple[str, ...],
) -> None:
    # Given source categories with searchable names and definitions.
    edit(memory, names=("retiring",))
    edit(memory, names=("other",))
    # When merge or split retires a source.
    edit(memory, operation, sources, tuple(1 for _ in sources), names)
    # Then neither lexical nor FTS search returns the retired definition.
    result = memory.call_tool(
        "search_memory", {"target": "categories", "mode": mode, "query": "retiring"}
    )
    assert result["total"] == 0
    assert memory.call_tool("read_categories", {"history": True, "category_ids": [1]})["total"] == 2


def test_exports_preserve_generic_raw_values_and_assignment_provenance(
    memory: TaxonomyMemory,
) -> None:
    # Given exact source strings assigned to an evidence-backed category.
    edit(memory)
    assign(memory)
    # When a downstream consumer exports assignments.
    assignments = memory.exports()["assignments"]
    # Then raw values and versioned decisions are usable without domain adapters.
    assert [row["raw_text"] for row in assignments] == [" raw ", "raw", "다른 원문"]
    assert all(
        row["category_name"] == "A"
        and row["category_version"] == 1
        and row["reason"] == "Definition matches this raw text"
        and "node_name" not in row
        for row in assignments
    )


def test_second_connection_rejects_an_obsolete_taxonomy_revision(tmp_path: Path) -> None:
    # Given two clients that opened the same persisted dataset.
    path = tmp_path / "shared.sqlite"
    with TaxonomyMemory(path) as writer, TaxonomyMemory(path) as reader:
        writer.ingest(["one"])
        reader.ingest(["one"])
        old_revision = reader.snapshot()["revision"]
        edit(writer)
        # When the second client submits its obsolete decision.
        with pytest.raises(MemoryConflict, match="Global revision"):
            reader.call_tool(
                "assign_items",
                {
                    "expected_revision": old_revision,
                    "assignments": [
                        {
                            "item_id": 1,
                            "category_id": 1,
                            "category_version": 1,
                            "reason": "Decision based on an obsolete catalog",
                        }
                    ],
                },
            )
        # Then the shared item remains pending with no assignment audit event.
        assert writer.snapshot()["pending"] == 1
        assert [change["kind"] for change in writer.exports()["changes"]] == ["create"]


def test_failed_edit_leaves_category_search_consistent(memory: TaxonomyMemory) -> None:
    # Given a searchable current category.
    edit(memory, names=("original",))
    # When a later edit in the same batch fails after the first changed FTS data.
    action = {
        "action": "revise",
        "category_ids": [1],
        "expected_versions": [1],
        "categories": [{"name": "replacement", "definition": "changed meaning"}],
        "evidence_ids": [1],
        "reason": "New evidence",
    }
    with pytest.raises(MemoryConflict, match="version conflict"):
        memory.call_tool("edit_categories", {"expected_revision": 1, "actions": [action, action]})
    # Then search observes the rolled-back definition, including its FTS index.
    original = memory.call_tool(
        "search_memory",
        {
            "target": "categories",
            "mode": "fts",
            "query": "original",
        },
    )
    replacement = memory.call_tool(
        "search_memory",
        {
            "target": "categories",
            "mode": "fts",
            "query": "replacement",
        },
    )
    assert (original["total"], replacement["total"]) == (1, 0)

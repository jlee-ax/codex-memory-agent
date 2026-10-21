"""Persistent, vector-free taxonomy memory; raw strings are never normalized."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from functools import wraps
from pathlib import Path
from types import TracebackType
from typing import Concatenate, Self

type JSON = str | int | float | bool | None | list[JSON] | dict[str, JSON]
type Record = dict[str, JSON]


class MemoryConflict(ValueError):
    """Invalid tool request or conflicting persisted state."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MemoryConflict(message)


def synchronized[**P, T](
    method: Callable[Concatenate[TaxonomyMemory, P], T],
) -> Callable[Concatenate[TaxonomyMemory, P], T]:
    @wraps(method)
    def guarded(self: TaxonomyMemory, *args: P.args, **kwargs: P.kwargs) -> T:
        with self._lock:
            return method(self, *args, **kwargs)

    return guarded


class TaxonomyMemory:
    """Mutable SQLite repository; callers own its lifetime via close/context manager."""

    def __init__(self, path: str | Path) -> None:
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1),
                fingerprint TEXT, revision INTEGER NOT NULL DEFAULT 0,
                stage TEXT NOT NULL DEFAULT 'building');
            INSERT OR IGNORE INTO state(id) VALUES(1);
            CREATE TABLE IF NOT EXISTS items (item_id INTEGER PRIMARY KEY, raw_text TEXT UNIQUE NOT NULL,
                category_id INTEGER, category_version INTEGER, reason TEXT,
                stale INTEGER NOT NULL DEFAULT 0, audit_revision INTEGER NOT NULL DEFAULT -1);
            CREATE TABLE IF NOT EXISTS categories (category_id INTEGER PRIMARY KEY,
                version INTEGER NOT NULL, name TEXT NOT NULL, definition TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1, evidence_ids TEXT NOT NULL DEFAULT '[]');
            CREATE TABLE IF NOT EXISTS category_versions (category_id INTEGER NOT NULL,
                version INTEGER NOT NULL, name TEXT NOT NULL, definition TEXT NOT NULL,
                active INTEGER NOT NULL, revision INTEGER NOT NULL, evidence_ids TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY(category_id,version));
            CREATE TABLE IF NOT EXISTS changes (change_id INTEGER PRIMARY KEY,
                revision INTEGER NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS item_search USING fts5(raw_text);
            CREATE VIRTUAL TABLE IF NOT EXISTS category_search USING fts5(name,definition);
            CREATE TRIGGER IF NOT EXISTS category_search_insert AFTER INSERT ON categories BEGIN
                INSERT INTO category_search(rowid,name,definition) VALUES(new.category_id,new.name,new.definition);
            END;
            CREATE TRIGGER IF NOT EXISTS category_search_update AFTER UPDATE ON categories BEGIN
                DELETE FROM category_search WHERE rowid=old.category_id;
                INSERT INTO category_search(rowid,name,definition) VALUES(new.category_id,new.name,new.definition);
            END;
            INSERT INTO category_search(rowid,name,definition)
                SELECT category_id,name,definition FROM categories
                WHERE category_id NOT IN (SELECT rowid FROM category_search);
        """)
        for table in ("categories", "category_versions"):
            if "evidence_ids" not in [
                row[1] for row in self.db.execute(f"PRAGMA table_info({table})")
            ]:
                self.db.execute(
                    f"ALTER TABLE {table} ADD COLUMN evidence_ids TEXT NOT NULL DEFAULT '[]'"
                )

    @synchronized
    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _rows(self, sql: str, params: tuple[JSON, ...] = ()) -> list[Record]:
        rows = [dict(row) for row in self.db.execute(sql, params)]
        for row in rows:
            if "evidence_ids" in row:
                row["evidence_ids"] = json.loads(row["evidence_ids"])
        return rows

    def _state(self) -> Record:
        return self._rows("SELECT * FROM state")[0]

    @synchronized
    def ingest(self, texts: Iterable[str]) -> Record:
        """Persist unique exact strings, or resume the same fingerprinted dataset."""
        raw = list(texts)
        require(all(isinstance(text, str) for text in raw), "All raw values must be strings")
        unique = sorted(set(raw))
        fingerprint = hashlib.sha256(json.dumps(unique, ensure_ascii=False).encode()).hexdigest()
        with self.db:
            old = self._state()["fingerprint"]
            require(
                old is None or old == fingerprint,
                "Dataset fingerprint mismatch; use a new memory file",
            )
            if old is None:
                self.db.executemany(
                    "INSERT INTO items(raw_text) VALUES(?)",
                    ((text,) for text in unique),
                )
                self.db.execute(
                    "INSERT INTO item_search(rowid,raw_text) SELECT item_id,raw_text FROM items"
                )
                self.db.execute("UPDATE state SET fingerprint=?", (fingerprint,))
        return self.snapshot()

    @synchronized
    def snapshot(self) -> Record:
        """Read progress counters and the current taxonomy revision atomically."""
        state = self._state()
        counts = self._rows(
            """SELECT count(*) AS total,
            coalesce(sum(category_id IS NULL),0) AS pending,
            coalesce(sum(stale),0) AS stale,
            coalesce(sum(audit_revision != ?),0) AS needs_review FROM items""",
            (state["revision"],),
        )[0]
        counts.update(state)
        counts["active_categories"] = self.db.execute(
            "SELECT count(*) FROM categories WHERE active=1"
        ).fetchone()[0]
        counts["complete"] = state["stage"] == "finalized"
        counts.update(
            phase={
                "building": "discovery",
                "final_review": "review",
                "finalized": "complete",
            }[state["stage"]],
            assigned=counts["total"] - counts["pending"],
            reviewed=counts["total"] - counts["needs_review"],
            review_pending=counts["needs_review"],
            k=self.db.execute(
                "SELECT count(*) FROM categories c WHERE active=1 AND EXISTS "
                "(SELECT 1 FROM items i WHERE i.category_id=c.category_id)"
            ).fetchone()[0],
        )
        return counts

    @synchronized
    def exports(self) -> Record:
        """Read categories, definition history, audit events, and raw assignments."""
        return {
            table: self._rows(f"SELECT * FROM {table}")
            for table in ("categories", "category_versions", "changes")
        } | {
            "assignments": self._rows(
                "SELECT i.*,c.name AS category_name,i.category_version AS version FROM items i LEFT JOIN categories c ON i.category_id=c.category_id"
            )
        }

    @staticmethod
    def tools() -> list[Record]:
        """Return JSON tool definitions for a model or another tool client."""
        integer = {"type": "integer"}
        text = {"type": "string"}
        ids = {"type": "array", "items": integer}
        category = {
            "type": "object",
            "properties": {"name": text, "definition": text},
            "required": ["name", "definition"],
            "additionalProperties": False,
        }
        action = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "revise", "merge", "split"],
                },
                "category_ids": ids,
                "expected_versions": {"type": "array", "items": integer},
                "categories": {"type": "array", "items": category},
                "evidence_ids": ids,
                "reason": text,
            },
            "required": [
                "action",
                "category_ids",
                "expected_versions",
                "categories",
                "evidence_ids",
                "reason",
            ],
            "additionalProperties": False,
        }
        assignment = {
            "type": "object",
            "properties": {
                "item_id": integer,
                "category_id": integer,
                "category_version": integer,
                "reason": text,
            },
            "required": ["item_id", "category_id", "category_version", "reason"],
            "additionalProperties": False,
        }
        specs = [
            ("memory_status", "Read progress and current global revision.", {}, []),
            (
                "next_items",
                "Page raw items: work includes unseen/stale; review includes all not audited at current revision; all reads every item.",
                {
                    "queue": {"type": "string", "enum": ["work", "review", "all"]},
                    "offset": integer,
                    "limit": integer,
                },
                ["queue"],
            ),
            (
                "search_memory",
                "Search raw items (default target=items) or active category names/definitions (target=categories), using literal substring or FTS5 queries. Returns total and bounded page with next_offset.",
                {
                    "query": text,
                    "target": {"type": "string", "enum": ["items", "categories"]},
                    "mode": {"type": "string", "enum": ["literal", "fts"]},
                    "offset": integer,
                    "limit": integer,
                },
                ["query", "mode"],
            ),
            (
                "read_categories",
                "Read current categories, or all historical versions with history=true, including inactive targets. Empty IDs pages entire catalog.",
                {
                    "category_ids": ids,
                    "history": {"type": "boolean"},
                    "offset": integer,
                    "limit": integer,
                },
                [],
            ),
            (
                "edit_categories",
                "Atomic semantic edits. Create has no source IDs and one new category; revise has one source/one replacement; merge >=2 sources/one replacement; split one source/>=2 replacements. All need raw evidence IDs and reason; expected_versions parallel source IDs. Every edit resets final review globally and stales source memberships.",
                {
                    "expected_revision": integer,
                    "actions": {"type": "array", "items": action},
                },
                ["expected_revision", "actions"],
            ),
            (
                "assign_items",
                "Atomic decisions using exact current category versions. During final_review each decision audits its item at global revision. All items require exactly one active category.",
                {
                    "expected_revision": integer,
                    "assignments": {"type": "array", "items": assignment},
                },
                ["expected_revision", "assignments"],
            ),
            (
                "review_control",
                "begin requires zero pending/stale and resets all audits. finalize requires exhaustive current-revision audit; freezes writes. reopen explicitly resumes building.",
                {"action": {"type": "string", "enum": ["begin", "finalize", "reopen"]}},
                ["action"],
            ),
        ]
        return [
            {
                "type": "function",
                "name": name,
                "description": desc,
                "inputSchema": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                    "additionalProperties": False,
                },
            }
            for name, desc, props, required in specs
        ]

    def _page(self, args: Record) -> tuple[int, int]:
        limit, offset = args.get("limit", 50), args.get("offset", 0)
        require(type(limit) is int and 1 <= limit <= 200, "limit must be 1..200")
        require(type(offset) is int and offset >= 0, "offset must be nonnegative")
        return limit, offset

    def _select(self, query: str, params: tuple[JSON, ...], args: Record) -> Record:
        limit, offset = self._page(args)
        total = self.db.execute(f"SELECT count(*) FROM ({query})", params).fetchone()[0]
        rows = self._rows(query + " LIMIT ? OFFSET ?", params + (limit, offset))
        page = {
            "total": total,
            "offset": offset,
            "rows": rows,
            "next_offset": None,
            "oversized": False,
        }
        while True:
            page["next_offset"] = offset + len(rows) if offset + len(rows) < total else None
            if len(json.dumps(page, ensure_ascii=True)) <= 60_000:
                return page
            if len(rows) <= 1:
                page["oversized"] = True
                return page
            rows.pop()

    @synchronized
    def call_tool(self, name: str, arguments: Record) -> Record:
        """Dispatch JSON tool requests; invalid writes raise and roll back entirely."""
        require(isinstance(arguments, dict), "arguments must be an object")
        require(self._state()["fingerprint"] is not None, "Call ingest first")
        match name:  # Untrusted external tool names are open strings.
            case "memory_status":
                return self.snapshot()
            case "next_items":
                filters = {
                    "work": "category_id IS NULL OR stale=1",
                    "review": "audit_revision != (SELECT revision FROM state)",
                    "all": "1",
                }
                require(arguments.get("queue") in filters, "Unknown queue")
                return self._select(
                    "SELECT * FROM items WHERE "
                    + filters[arguments["queue"]]
                    + " ORDER BY item_id",
                    (),
                    arguments,
                )
            case "search_memory":
                require(isinstance(arguments.get("query"), str), "query must be text")
                queries = {
                    (
                        "items",
                        "literal",
                    ): "SELECT * FROM items WHERE instr(raw_text,?1)>0 ORDER BY item_id",
                    (
                        "items",
                        "fts",
                    ): "SELECT * FROM items WHERE item_id IN (SELECT rowid FROM item_search WHERE item_search MATCH ?1) ORDER BY item_id",
                    (
                        "categories",
                        "literal",
                    ): "SELECT * FROM categories WHERE active=1 AND (instr(name,?1)>0 OR instr(definition,?1)>0) ORDER BY category_id",
                    (
                        "categories",
                        "fts",
                    ): "SELECT * FROM categories WHERE active=1 AND category_id IN (SELECT rowid FROM category_search WHERE category_search MATCH ?1) ORDER BY category_id",
                }
                target, mode = arguments.get("target", "items"), arguments.get("mode")
                require(
                    isinstance(target, str) and isinstance(mode, str) and (target, mode) in queries,
                    "Unknown search target or mode",
                )
                return self._select(queries[target, mode], (arguments["query"],), arguments)
            case "read_categories":
                ids = arguments.get("category_ids", [])
                require(
                    isinstance(ids, list) and all(type(i) is int for i in ids),
                    "category_ids must be integers",
                )
                where = " WHERE category_id IN (" + ",".join("?" for _ in ids) + ")" if ids else ""
                require(
                    type(arguments.get("history", False)) is bool,
                    "history must be boolean",
                )
                table = "category_versions" if arguments.get("history", False) else "categories"
                return self._select(
                    "SELECT * FROM " + table + where + " ORDER BY category_id,version",
                    tuple(ids),
                    arguments,
                )
            case "edit_categories" | "assign_items" | "review_control":
                with self.db:
                    self.db.execute("BEGIN IMMEDIATE")
                    state = self._state()
                    match name:
                        case "review_control":
                            self._review(arguments)
                        case "edit_categories":
                            self._writable(arguments, state)
                            actions = arguments.get("actions")
                            require(
                                isinstance(actions, list) and bool(actions),
                                "actions must be nonempty",
                            )
                            for action in actions:
                                self._edit(action)
                        case "assign_items":
                            self._writable(arguments, state)
                            self._assign(arguments, state)
                return self.snapshot()
            case _:
                raise MemoryConflict("Unknown memory tool: " + name)

    def _writable(self, args: Record, state: Record) -> None:
        require(
            state["stage"] != "finalized",
            "Finalized memory is frozen; explicitly reopen first",
        )
        require(
            type(args.get("expected_revision")) is int
            and args["expected_revision"] == state["revision"],
            "Global revision conflict",
        )

    def _category(self, category_id: int, version: int) -> Record:
        rows = self._rows(
            "SELECT * FROM categories WHERE category_id=? AND version=? AND active=1",
            (category_id, version),
        )
        require(bool(rows), "Inactive/missing category or version conflict")
        return rows[0]

    def _history(self, kind: str, payload: Record) -> None:
        self.db.execute(
            "INSERT INTO changes(revision,kind,payload) VALUES((SELECT revision FROM state),?,?)",
            (kind, json.dumps(payload, ensure_ascii=False)),
        )

    def _save_version(self, category_id: int) -> None:
        self.db.execute(
            "INSERT INTO category_versions SELECT category_id,version,name,definition,active,(SELECT revision FROM state),evidence_ids FROM categories WHERE category_id=?",
            (category_id,),
        )

    def _edit(self, action: Record) -> None:
        require(isinstance(action, dict), "action must be an object")
        ids, versions, categories = (
            action.get("category_ids"),
            action.get("expected_versions"),
            action.get("categories"),
        )
        require(
            isinstance(ids, list) and isinstance(versions, list) and len(ids) == len(versions),
            "Source IDs/versions must be parallel lists",
        )
        require(
            all(type(i) is int for i in ids + versions) and len(set(ids)) == len(ids),
            "Source IDs/versions must be unique integer IDs",
        )
        require(isinstance(categories, list), "categories must be a list")
        require(
            isinstance(action.get("reason"), str) and bool(action["reason"].strip()),
            "reason required",
        )
        evidence = action.get("evidence_ids")
        require(
            isinstance(evidence, list) and bool(evidence) and all(type(i) is int for i in evidence),
            "Raw evidence IDs required",
        )
        for item_id in evidence:
            require(
                self.db.execute("SELECT 1 FROM items WHERE item_id=?", (item_id,)).fetchone()
                is not None,
                "Unknown evidence item",
            )
        for category in categories:
            require(
                isinstance(category, dict)
                and all(
                    isinstance(category.get(k), str) and category[k].strip()
                    for k in ("name", "definition")
                ),
                "Category name and definition required",
            )
        for category_id, version in zip(ids, versions, strict=True):
            self._category(category_id, version)
        kind = action.get("action")
        shapes = {
            "create": len(ids) == 0 and len(categories) == 1,
            "revise": len(ids) == 1 and len(categories) == 1,
            "merge": len(ids) >= 2 and len(categories) == 1,
            "split": len(ids) == 1 and len(categories) >= 2,
        }
        require(kind in shapes and shapes[kind], "Invalid edit action shape")
        self.db.execute("UPDATE state SET revision=revision+1,stage='building'")
        for category_id in ids:
            self.db.execute("UPDATE items SET stale=1 WHERE category_id=?", (category_id,))
        match kind:  # Action names were validated above.
            case "revise":
                self.db.execute(
                    "UPDATE categories SET version=version+1,name=?,definition=?,evidence_ids=? WHERE category_id=?",
                    (
                        categories[0]["name"],
                        categories[0]["definition"],
                        json.dumps(evidence),
                        ids[0],
                    ),
                )
                self._save_version(ids[0])
            case "create" | "merge" | "split":
                for category_id in ids:
                    self.db.execute(
                        "UPDATE categories SET active=0,version=version+1 WHERE category_id=?",
                        (category_id,),
                    )
                    self._save_version(category_id)
                for category in categories:
                    cursor = self.db.execute(
                        "INSERT INTO categories(version,name,definition,evidence_ids) VALUES(1,?,?,?)",
                        (
                            category["name"],
                            category["definition"],
                            json.dumps(evidence),
                        ),
                    )
                    self._save_version(cursor.lastrowid)
        self._history(str(kind), action)

    def _assign(self, args: Record, state: Record) -> None:
        assignments = args.get("assignments")
        require(
            isinstance(assignments, list) and bool(assignments),
            "assignments must be nonempty",
        )
        seen: set[int] = set()
        for assignment in assignments:
            require(isinstance(assignment, dict), "assignment must be an object")
            require(
                all(
                    type(assignment.get(k)) is int
                    for k in ("item_id", "category_id", "category_version")
                ),
                "Assignment IDs/versions must be integers",
            )
            require(
                isinstance(assignment.get("reason"), str) and bool(assignment["reason"].strip()),
                "Assignment reason required",
            )
            item_id = assignment["item_id"]
            require(item_id not in seen, "Duplicate item in assignment batch")
            seen.add(item_id)
            self._category(assignment["category_id"], assignment["category_version"])
            cursor = self.db.execute(
                "UPDATE items SET category_id=?,category_version=?,reason=?,stale=0,audit_revision=? WHERE item_id=?",
                (
                    assignment["category_id"],
                    assignment["category_version"],
                    assignment["reason"],
                    state["revision"] if state["stage"] == "final_review" else -1,
                    item_id,
                ),
            )
            require(cursor.rowcount == 1, "Unknown item ID")
            self._history("assign", assignment)

    def _review(self, args: Record) -> None:
        status = self.snapshot()
        match args.get("action"):  # Boundary validation for external input.
            case "reopen":
                self.db.execute("UPDATE state SET stage='building'")
                self.db.execute("UPDATE items SET audit_revision=-1")
            case "begin" | "finalize":
                require(status["stage"] != "finalized", "Already finalized; reopen first")
                require(
                    status["pending"] == 0 and status["stale"] == 0,
                    "Unassigned or stale items remain",
                )
                invalid = self.db.execute(
                    "SELECT count(*) FROM items i LEFT JOIN categories c ON i.category_id=c.category_id WHERE c.active IS NOT 1 OR i.category_version != c.version"
                ).fetchone()[0]
                require(invalid == 0, "Invalid category references remain")
                if args["action"] == "finalize":
                    require(
                        status["stage"] == "final_review" and status["needs_review"] == 0,
                        "Exhaustive final review required",
                    )
                    self.db.execute("UPDATE state SET stage='finalized'")
                else:
                    self.db.execute("UPDATE state SET stage='final_review'")
                    self.db.execute("UPDATE items SET audit_revision=-1")
            case _:
                raise MemoryConflict("Unknown review action")
        self._history("review", args)

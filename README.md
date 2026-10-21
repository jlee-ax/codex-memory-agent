# codex-memory-agent

Connect Python or Jupyter to your local Codex account session, and let an agent build a taxonomy in persistent external memory. No embedding model, vector database, fixed category count, or dataset-specific schema is required.

This project provides two independently usable layers:

| Layer | Public interface | Purpose |
| --- | --- | --- |
| Connection | `CodexConnection` | Run Codex turns from Python and handle application-defined tools over JSON-RPC. |
| Classification | `TaxonomyAgent`, `TaxonomyMemory` | Read original text in batches, evolve category definitions, persist assignments, and review every item against the final taxonomy. |

Core Python dependencies: **none**. Python 3.12+ and a SQLite build with FTS5 are required. Notebook progress uses optional IPython and tqdm dependencies.

[한국어 안내](README.ko.md) · [Architecture and protocol](docs/architecture.md) · [Examples](examples)

## Install and authenticate

Install the [Codex CLI](https://learn.chatgpt.com/docs/cli) separately, then sign in with your own ChatGPT account:

```bash
codex login
codex --version
python -m pip install 'git+https://github.com/jlee-ax/codex-memory-agent.git'
```

For a Jupyter kernel, use its own package installer:

```python
%pip install "codex-memory-agent[notebook] @ git+https://github.com/jlee-ax/codex-memory-agent.git"
```

The package invokes `codex app-server` and checks managed ChatGPT authentication through `account/read`. It does not copy browser cookies, extract tokens, create credentials, or require an OpenAI API key. Codex CLI must be available on the kernel's PATH; an explicit executable command is also supported.

This integration targets local, open-source applications, with each person using their own account. OpenAI's [App Server documentation](https://learn.chatgpt.com/docs/app-server) describes this authentication scope and recommends Sign in with ChatGPT for new integrations; managed App Server authentication is not a hosted or commercial service backend. Account eligibility and usage limits still apply. The App Server dynamic tools interface is experimental and may change.

## Classify your own text

```python
from codex_memory_agent import TaxonomyAgent

texts = [
    "Draft a project proposal",
    "Summarize customer feedback",
    "Debug a Python function",
    "Write a unit test",
]
prompt = """
Group these tasks by the underlying work performed. Start with no categories.
Create, revise, merge, or split categories when the original evidence warrants it.
Choose the number of categories from the data. Use clear inclusion and exclusion
criteria, and review all items against the final definitions.
"""

with TaxonomyAgent() as agent:
    result = agent.run(texts, prompt, directory=".codex-memory-agent/demo")
    print("Categories:", result["k"])
    print(result["clusters"])
    print(result["assignments"])
```

Each distinct original string receives one stable item ID and one category assignment. Exact duplicate strings are classified once. Whitespace, case, and punctuation are preserved. If your table contains repeated rows, join the returned `raw_text` assignments back to the original table.

`result` contains `clusters` (occupied, active categories), `k`, `assignments`, `categories`, `category_versions`, `changes`, and `status`. Assignments include `raw_text`, category ID, current definition version, and a decision reason. Treat results as provisional unless `result["status"]["complete"]` is true.

## Jupyter: start, monitor, pause, resume

Keep the agent alive across cells:

```python
from codex_memory_agent import TaxonomyAgent

agent = TaxonomyAgent()
agent.start(texts, prompt, directory=".codex-memory-agent/demo", batch_size=40)
agent.monitor()
result = agent.results()
```

`monitor()` displays initial coverage and final review progress. Ctrl-C pauses this job; committed decisions remain in SQLite. Alternatively call `agent.stop()`. Inspect `agent.status()` and resume with the same `start(...)` call. When finished, call `agent.close()` to release its worker and owned Codex process. Closing this agent does not stop your Jupyter kernel.

The same original strings, exact prompt, and directory resume the same database after a Python or kernel restart. Changing the prompt or data creates a separate database. `batch_size` is a per-turn instruction for context management, not a dataset limit or a hard tool-call quota. Category count is not capped. `turn_timeout` bounds one model turn; `wait(timeout=...)` bounds the caller's wait without cancelling the job.

## Use only the connection layer

```python
from codex_memory_agent import CodexConnection

with CodexConnection() as connection:
    answer = connection.run_turn(
        "Explain what a persistent memory store contributes to a local agent.",
        dynamic_tools=[],
        tool_handler=lambda name, arguments: {"error": f"Unknown tool: {name}"},
    )
    print(answer)
```

Applications can provide JSON-schema tool specifications and a `tool_handler(name, arguments)` callback. See [custom_tools.py](examples/custom_tools.py) for a complete example. An externally supplied connection remains open when its `TaxonomyAgent` closes, unless `own_connection=True` is set.

```python
connection = CodexConnection(command=["/path/to/codex", "app-server"])
```

## Memory and completion semantics

The SQLite database is the authoritative state. Each turn starts with a compact status and retrieves originals, category definitions, or evidence through tools. Category edits use revision checks, preserve definition history, and invalidate affected assignments. A separate final review must cover every item at the current taxonomy revision before the database can be finalized.

Literal and SQLite FTS5 search retrieve text; they do not produce embeddings. The model makes semantic classification decisions. Retrieval reduces repeated context, but reading and reviewing the complete dataset still requires model work proportional to the number of items. This library does not train model weights or guarantee optimal categories, accuracy, convergence, or a particular throughput.

Inputs are short text records. Whole records are returned without truncation; a single unusually long record can exceed the model context. Split long documents into meaningful records before ingestion. SQLite memory and exports contain your original text. Tool handlers run in your Python process, so only expose operations appropriate for your application. The read-only Codex sandbox and prompt policy do not make arbitrary custom tools safe.

## Development

```bash
git clone https://github.com/jlee-ax/codex-memory-agent.git
cd codex-memory-agent
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check .
python -m build
```

Tests use a fake JSON-RPC subprocess and temporary SQLite databases; they do not require a Codex login or make model requests. CI runs these checks on Python 3.12, 3.13, and 3.14. The initial live compatibility check used Codex CLI `0.163.0-alpha.10.2` on macOS; other CLI releases and operating systems need their own live verification.

MIT licensed. This is an independent community project.

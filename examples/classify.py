from codex_memory_agent import TaxonomyAgent

TEXTS = [
    "Draft a project proposal",
    "Summarize customer feedback",
    "Debug a Python function",
    "Write a unit test",
]
PROMPT = """Group tasks by the work performed. Begin with no categories and let k
emerge from the originals. Maintain clear inclusion and exclusion criteria.
Revise, merge, or split categories when evidence requires it. Review every item
against the final definitions."""


def main() -> None:
    with TaxonomyAgent() as agent:
        result = agent.run(TEXTS, PROMPT, directory=".codex-memory-agent/example")
        print("Complete:", result["status"]["complete"])
        print("Categories:", result["k"])
        print(result["assignments"])


if __name__ == "__main__":
    main()

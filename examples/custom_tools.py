from codex_memory_agent import CodexConnection
from codex_memory_agent.transport import Message

TOOLS: list[Message] = [
    {
        "type": "function",
        "name": "project_glossary",
        "description": "Read the project's definition of external memory.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    }
]


def handle_tool(name: str, arguments: Message) -> Message:
    if name != "project_glossary" or arguments:
        return {"error": "Unknown tool or unexpected arguments"}
    return {"definition": "External memory is persistent application state retrieved through tools."}


def main() -> None:
    with CodexConnection() as connection:
        answer = connection.run_turn(
            "Use project_glossary and explain its definition in one sentence.",
            dynamic_tools=TOOLS,
            tool_handler=handle_tool,
        )
        print(answer)


if __name__ == "__main__":
    main()

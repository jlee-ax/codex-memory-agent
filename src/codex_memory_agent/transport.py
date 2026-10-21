"""Account-authenticated, synchronous Codex App Server JSON-RPC transport."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Self

type JSON = None | bool | int | float | str | list[JSON] | dict[str, JSON]
type Message = dict[str, JSON]


class CodexError(RuntimeError):
    """A transport or protocol failure with no server secrets in its message."""


class CodexRPCError(CodexError):
    def __init__(self, method: str, code: JSON) -> None:
        self.code = code
        super().__init__(f"{method} failed (RPC code {code})")


class TurnCancelled(CodexError, InterruptedError):
    """The caller cancelled a turn."""


def _mapping(value: JSON) -> Message:
    if not isinstance(value, dict):
        raise CodexError("Expected a JSON object")
    return value


def _identifier(value: JSON) -> str:
    if not isinstance(value, str) or not value:
        raise CodexError("Missing protocol identifier")
    return value


class CodexConnection:
    """Own one subprocess; mutate queues to route concurrent RPC responses safely."""

    def __init__(
        self,
        command: Sequence[str] | None = None,
        cwd: str | None = None,
        timeout: float = 30,
        *,
        client_info: Message | None = None,
    ) -> None:
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._turn_lock = threading.Lock()
        self._pending: dict[int, queue.Queue[Message | CodexError]] = {}
        self._events: queue.Queue[Message | CodexError] = queue.Queue()
        self._next_id = 0
        self._failure: CodexError | None = None
        self._closed = False
        self._stderr: deque[str] = deque(maxlen=32)
        self.cleanup_errors: list[str] = []
        try:
            self._process = subprocess.Popen(
                list(command) if command is not None else ["codex", "app-server"],
                cwd=cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise CodexError(
                "Codex CLI or working directory was not found. Install the Codex CLI, "
                "make it available on PATH, or supply command with its executable path."
            ) from exc
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._err_reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._reader.start()
        self._err_reader.start()
        try:
            self.request(
                "initialize",
                {
                    "clientInfo": client_info
                    if client_info is not None
                    else {
                        "name": "codex-memory-agent",
                        "title": "Codex Memory Agent",
                        "version": "0.1.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
                timeout,
            )
            self._send({"method": "initialized"})
            account = self.request("account/read", {"refreshToken": False}, timeout)
            if _mapping(account.get("account")).get("type") != "chatgpt":
                raise CodexError("Existing ChatGPT account authentication is required")
        except BaseException:  # Release the owned process, then propagate the original failure.
            self.close()
            raise

    def _send(self, message: Message) -> None:
        with self._write_lock:
            if self._closed or self._process.stdin is None:
                raise CodexError("Connection is closed")
            try:
                self._process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise CodexError("App Server input closed") from exc

    def _fail(self, failure: CodexError) -> None:
        with self._state_lock:
            self._failure = failure
            for response in self._pending.values():
                response.put(failure)
        self._events.put(failure)

    def _read(self) -> None:
        try:
            assert self._process.stdout is not None
            for line in self._process.stdout:
                message = _mapping(json.loads(line))
                if "method" in message:
                    if "id" in message and message["method"] != "item/tool/call":
                        self._send(
                            {
                                "id": message["id"],
                                "error": {
                                    "code": -32601,
                                    "message": "Client does not support this request",
                                },
                            }
                        )
                    else:
                        self._events.put(message)
                else:
                    with self._state_lock:
                        response_id = message.get("id")
                        if not isinstance(response_id, int) or isinstance(response_id, bool):
                            raise CodexError("Invalid response identifier")
                        response = self._pending.get(response_id)
                    if response is not None:
                        response.put(message)
        except (OSError, ValueError, CodexError):
            self._fail(CodexError("App Server protocol stream failed"))
        finally:
            self._fail(CodexError("App Server exited"))

    def _drain_stderr(self) -> None:
        assert self._process.stderr is not None
        while chunk := self._process.stderr.read(1024):
            self._stderr.append(chunk)  # Private, bounded, never included in errors/logs.

    def request(self, method: str, params: Message, timeout: float = 30) -> Message:
        response: queue.Queue[Message | CodexError] = queue.Queue()
        with self._state_lock:
            if self._failure is not None:
                raise self._failure
            self._next_id += 1
            request_id = self._next_id
            self._pending[request_id] = response
        try:
            self._send({"id": request_id, "method": method, "params": params})
            try:
                result = response.get(timeout=max(0, timeout))
            except queue.Empty:
                raise TimeoutError(f"{method} timed out") from None
            if isinstance(result, CodexError):
                raise result
            if "error" in result:
                raise CodexRPCError(method, _mapping(result["error"]).get("code"))
            return _mapping(result.get("result"))
        finally:
            with self._state_lock:
                self._pending.pop(request_id, None)

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
        """Run one fresh thread; handlers must be bounded local operations."""
        if not self._turn_lock.acquire(blocking=False):
            raise CodexError("A turn is already running on this connection")
        thread_id = turn_id = None
        completed = False
        deadline = time.monotonic() + timeout
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise TurnCancelled("Turn cancelled")
            params: Message = {
                "ephemeral": True,
                "sandbox": "read-only",
                "approvalPolicy": "never",
                "dynamicTools": list(dynamic_tools),
            }
            if model is not None:
                params["model"] = model
            thread_id = _identifier(
                _mapping(self.request("thread/start", params, min(30, timeout))["thread"])["id"]
            )
            started = self.request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": prompt, "text_elements": []}],
                },
                max(0, min(30, deadline - time.monotonic())),
            )
            turn_id = _identifier(_mapping(started["turn"])["id"])
            answers: dict[str, str] = {}
            final_answer: str | None = None
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise TurnCancelled("Turn cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Turn timed out")
                try:
                    message = self._events.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                if isinstance(message, CodexError):
                    raise message
                event = _mapping(message.get("params", {}))
                if event.get("threadId") != thread_id:
                    if "id" in message:
                        self._send(
                            {
                                "id": message["id"],
                                "error": {"code": -32602, "message": "Inactive thread"},
                            }
                        )
                    continue
                if on_event is not None:
                    on_event(message)
                match message["method"]:
                    case "item/tool/call":
                        name = _identifier(event.get("tool"))
                        if event.get("turnId") != turn_id or name not in [
                            tool.get("name") for tool in dynamic_tools
                        ]:
                            self._send(
                                {
                                    "id": message["id"],
                                    "error": {
                                        "code": -32602,
                                        "message": "Inactive turn or unregistered tool",
                                    },
                                }
                            )
                            continue
                        try:
                            output = tool_handler(name, _mapping(event.get("arguments")))
                            text = (
                                output
                                if isinstance(output, str)
                                else json.dumps(output, ensure_ascii=False)
                            )
                            success = True
                        except Exception as exc:  # Redact tool failures before transmission.
                            text, success = f"Local tool failed: {type(exc).__name__}", False
                        self._send(
                            {
                                "id": message["id"],
                                "result": {
                                    "success": success,
                                    "contentItems": [{"type": "inputText", "text": text}],
                                },
                            }
                        )
                    case "item/completed":
                        item = _mapping(event.get("item"))
                        if item.get("type") == "agentMessage":
                            answer = _identifier(item.get("text"))
                            answers[_identifier(item.get("id"))] = answer
                            if item.get("phase") == "final_answer":
                                final_answer = answer
                    case "turn/completed":
                        turn = _mapping(event.get("turn"))
                        if turn.get("id") != turn_id:
                            continue
                        completed = True
                        if turn.get("status") != "completed":
                            raise CodexError(f"Turn ended with status {turn.get('status')}")
                        return (
                            final_answer
                            if final_answer is not None
                            else next(reversed(answers.values()), "")
                        )
        finally:
            if thread_id is not None:
                cleanup: list[tuple[str, Message]] = [
                    ("thread/unsubscribe", {"threadId": thread_id})
                ]
                if turn_id is not None and not completed:
                    cleanup.insert(
                        0, ("turn/interrupt", {"threadId": thread_id, "turnId": turn_id})
                    )
                for method, params in cleanup:
                    try:
                        self.request(method, params, 5)
                    except (CodexError, TimeoutError) as exc:
                        self.cleanup_errors.append(str(exc))
                        del self.cleanup_errors[:-50]
                if turn_id is None and not completed:
                    self.close()  # A lost turn/start response leaves the turn id unknowable.
            self._turn_lock.release()

    def close(self) -> None:
        """Terminate and reap only this connection's subprocess."""
        with self._write_lock:
            if self._closed:
                return
            self._closed = True
            if self._process.poll() is None:
                self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        self._reader.join(timeout=5)
        self._err_reader.join(timeout=5)
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                stream.close()
        self._stderr.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

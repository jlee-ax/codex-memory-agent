"""Wire-level tests use an isolated child process, never a real model."""

import sys
import threading
import unittest
from typing import Final

from codex_memory_agent.transport import (
    JSON,
    CodexConnection,
    CodexError,
    CodexRPCError,
    Message,
    TurnCancelled,
)

SERVER: Final = r"""
import json,sys
mode=sys.argv[1]
count=0
loaded=set()
interrupted=[]
for raw in sys.stdin:
    m=json.loads(raw)
    method=m.get('method')
    if method is None:
        if m['id']==99:
            assert m['result']['contentItems'][0]['type']=='inputText'
            expected=mode!='tool_error'
            assert m['result']['success']==expected
            print(json.dumps({'method':'item/completed','params':{'threadId':tid,'item':{'id':'comment','type':'agentMessage','phase':'commentary','text':'working'}}}),flush=True)
            print(json.dumps({'method':'item/completed','params':{'threadId':tid,'item':{'id':'answer','type':'agentMessage','phase':'final_answer','text':'tool response received'}}}),flush=True)
            print(json.dumps({'method':'turn/completed','params':{'threadId':tid,'turn':{'id':'turn','status':'completed'}}}),flush=True)
        elif m['id']==98:
            assert m['error']['code']==-32601
        continue
    if 'id' not in m: continue
    p=m.get('params',{})
    result={}
    if method=='initialize':
        assert p['capabilities']['experimentalApi'] is True
        client_info=p['clientInfo']
    elif method=='account/read': result={'account':{'type':'apiKey' if mode=='api_key' else 'chatgpt'}}
    elif method=='thread/start':
        assert p['ephemeral'] and p['sandbox']=='read-only' and p['approvalPolicy']=='never'
        count+=1;tid='thread'+str(count);loaded.add(tid);result={'thread':{'id':tid}}
    elif method=='turn/start':
        if mode=='eof': sys.exit(0)
        result={'turn':{'id':'turn'}}
    elif method=='echo': result=p
    elif method=='client_info': result=client_info
    elif method=='rpc_error':
        print(json.dumps({'id':m['id'],'error':{'code':-32001,'message':'secret'}}),flush=True)
        continue
    elif method=='bad_id':
        print(json.dumps({'id':[],'result':{}}),flush=True)
        continue
    elif method=='exit': sys.exit(0)
    elif method=='stall': continue
    elif method=='turn/interrupt': interrupted.append(p['threadId'])
    elif method=='thread/unsubscribe':
        if mode=='cleanup_error':
            print(json.dumps({'id':m['id'],'error':{'code':-32600,'message':'failed'}}),flush=True)
            continue
        assert p['threadId'] in loaded
        loaded.remove(p['threadId'])
        result={'status':'unsubscribed'}
    elif method=='lifecycle': result={'loaded':sorted(loaded),'interrupted':interrupted}
    elif method=='thread/archive': raise AssertionError('Ephemeral threads must be unsubscribed')
    print(json.dumps({'id':m['id'],'result':result}),flush=True)
    if method=='turn/start' and mode=='active_eof': sys.exit(0)
    if method=='turn/start' and mode not in ('idle','eof'):
        print(json.dumps({'id':98,'method':'item/commandExecution/requestApproval','params':{'threadId':tid}}),flush=True)
        print(json.dumps({'id':99,'method':'item/tool/call','params':{'threadId':tid,'turnId':'turn','callId':'call','tool':'lookup','arguments':{'query':'hello'}}}),flush=True)
"""
TOOLS: Final[list[Message]] = [
    {
        "type": "function",
        "name": "lookup",
        "description": "lookup",
        "inputSchema": {"type": "object"},
    }
]


class TransportTests(unittest.TestCase):
    def connect(self, mode: str = "normal") -> CodexConnection:
        return CodexConnection([sys.executable, "-u", "-c", SERVER, mode], timeout=2)

    def test_dynamic_tool_and_nested_rpc_when_reused(self) -> None:
        # Given a wire server with callback and unsupported approval requests.
        with self.connect() as connection:
            calls: list[tuple[str, Message]] = []

            def handler(name: str, args: Message) -> JSON:
                calls.append((name, args))
                return connection.request("echo", args, 1)

            # When two sequential ephemeral turns call a tool that itself makes RPC.
            results = [connection.run_turn("test", TOOLS, handler, timeout=3) for _ in range(2)]
            # Then reader is free, callbacks succeed, and the object is reusable.
            self.assertEqual(results, ["tool response received"] * 2)
            self.assertEqual(calls, [("lookup", {"query": "hello"})] * 2)
            self.assertEqual(
                connection.request("lifecycle", {}, 1), {"loaded": [], "interrupted": []}
            )

    def test_tool_exception_when_handler_fails(self) -> None:
        # Given a failing local tool.
        with self.connect("tool_error") as connection:

            def handler(name: str, args: Message) -> JSON:
                raise CodexError("private information")

            # When server receives a failed tool response.
            result = connection.run_turn("test", TOOLS, handler, timeout=3)
            # Then turn can finish with the tool error represented in protocol.
            self.assertEqual(result, "tool response received")

    def test_eof_when_turn_start_pending(self) -> None:
        # Given a child that exits during turn/start.
        with self.connect("eof") as connection, self.assertRaisesRegex(CodexError, "exited"):
            # When pending RPC loses its server, then it wakes with explicit EOF.
            connection.run_turn("test", TOOLS, lambda *_: {}, timeout=3)

    def test_eof_when_turn_active(self) -> None:
        # Given a child that exits after acknowledging turn/start.
        with self.connect("active_eof") as connection, self.assertRaisesRegex(CodexError, "exited"):
            # When the active event loop loses the child, then EOF is propagated.
            connection.run_turn("test", TOOLS, lambda *_: {}, timeout=3)

    def test_timeout_when_turn_active(self) -> None:
        # Given a child that acknowledges start but never completes.
        with self.connect("idle") as connection:
            # When the turn deadline expires, then interrupt/unsubscribe finish.
            with self.assertRaises(TimeoutError):
                connection.run_turn("test", TOOLS, lambda *_: {}, timeout=0.1)
            self.assertEqual(connection.cleanup_errors, [])
            self.assertEqual(
                connection.request("lifecycle", {}, 1), {"loaded": [], "interrupted": ["thread1"]}
            )

    def test_cancel_when_active_turn(self) -> None:
        # Given an active turn signalled through a deterministic RPC boundary.
        with self.connect("idle") as connection:
            cancel = threading.Event()
            original = connection.request

            def request(method: str, params: Message, timeout: float = 30) -> Message:
                result = original(method, params, timeout)
                if method == "turn/start":
                    cancel.set()
                return result

            connection.request = request
            # When cancellation is raised, then interrupt/unsubscribe complete.
            with self.assertRaises(TurnCancelled):
                connection.run_turn("test", TOOLS, lambda *_: {}, cancel_event=cancel)
            self.assertEqual(connection.cleanup_errors, [])
            self.assertEqual(
                connection.request("lifecycle", {}, 1), {"loaded": [], "interrupted": ["thread1"]}
            )

    def test_timeout_when_rpc_stalls(self) -> None:
        # Given a server that intentionally ignores one request.
        with self.connect() as connection:
            # When timeout expires, then no indefinitely pending call remains.
            with self.assertRaises(TimeoutError):
                connection.request("stall", {}, 0.05)
            self.assertEqual(connection.request("echo", {"ok": True}, 1), {"ok": True})

    def test_auth_when_api_key_account(self) -> None:
        # Given an API-key account, when constructing, then reject it.
        with self.assertRaisesRegex(CodexError, "ChatGPT"):
            self.connect("api_key")

    def test_cleanup_errors_when_repeated_failures(self) -> None:
        # Given a server that rejects unsubscribe requests.
        with self.connect("cleanup_error") as connection:
            # When many sequential turns encounter cleanup failures.
            for _ in range(55):
                connection.run_turn("test", TOOLS, lambda *_: {}, timeout=3)
            # Then retained diagnostics remain bounded and expose the failure.
            self.assertEqual(
                connection.cleanup_errors, ["thread/unsubscribe failed (RPC code -32600)"] * 50
            )

    def test_close_when_own_process(self) -> None:
        # Given our own running child.
        connection = self.connect()
        # When closed, then process and readers finish; repeated close is safe.
        connection.close()
        connection.close()
        self.assertIsNotNone(connection._process.poll())
        self.assertFalse(connection._reader.is_alive())
        self.assertFalse(connection._err_reader.is_alive())

    def test_missing_executable_when_command_unavailable(self) -> None:
        # Given an explicit command that cannot exist.
        # When launching it, then the transport exposes its public error type.
        with self.assertRaises(CodexError) as caught:
            CodexConnection(["/nonexistent/codex-memory-agent-cli"], timeout=1)
        self.assertIsInstance(caught.exception.__cause__, FileNotFoundError)

    def test_client_info_when_overridden(self) -> None:
        # Given application-specific metadata.
        expected: Message = {"name": "custom-client", "version": "9.0"}
        # When initializing a connection, then the server sees that metadata.
        with CodexConnection(
            [sys.executable, "-u", "-c", SERVER, "normal"], timeout=2, client_info=expected
        ) as connection:
            self.assertEqual(connection.request("client_info", {}, 1), expected)

    def test_rpc_error_when_server_rejects_request(self) -> None:
        # Given an initialized server.
        with self.connect() as connection:
            # When an RPC fails, then retain its code without leaking server prose.
            with self.assertRaises(CodexRPCError) as caught:
                connection.request("rpc_error", {}, 1)
            self.assertEqual(caught.exception.code, -32001)
            self.assertNotIn("secret", str(caught.exception))

    def test_protocol_error_when_response_id_is_malformed(self) -> None:
        # Given a server capable of sending malformed IDs.
        with self.connect() as connection:
            # When it emits an array ID, then wake the request with a protocol error.
            with self.assertRaises(CodexError):
                connection.request("bad_id", {}, 1)

    def test_close_when_another_connection_is_alive(self) -> None:
        # Given two independently owned child processes.
        with self.connect() as first, self.connect() as second:
            # When one closes, then the other still answers requests.
            first.close()
            self.assertEqual(second.request("echo", {"alive": True}, 1), {"alive": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)

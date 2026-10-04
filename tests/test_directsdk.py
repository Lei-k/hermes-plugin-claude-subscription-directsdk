import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import pytest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAKE = r"""
import json, os, pathlib, sys, time
if os.environ.get('PID_FILE'):
 open(os.environ['PID_FILE'],'w').write(str(os.getpid()))
if '--version' in sys.argv:
 print('2.1.263 (Claude Code)'); sys.exit()
rows=[]
for line in sys.stdin:
 r=json.loads(line); rows.append(r)
 if r.get('shouldQuery') is False:
  print(json.dumps({'type':'result','num_turns':0,'is_error':False}),flush=True)
if os.environ.get('NATIVE_ERROR'):
 # An error native answers itself (no login: authentication_failed), with no request to ANTHROPIC_BASE_URL.
 code,text=os.environ['NATIVE_ERROR'].split(':',1)
 print(json.dumps({'type':'assistant','error':code,'is_api_error_message':True,'message':{'role':'assistant','model':'<synthetic>','content':[{'type':'text','text':text}],'stop_reason':'stop_sequence'}}),flush=True)
 print(json.dumps({'type':'result','subtype':'success','is_error':True,'num_turns':1,'result':text}),flush=True)
 sys.exit(1)
if os.environ.get('HANG'):
 if os.environ.get('PID_FILE'):
  # Real native is a shim -> node tree; cancellation must take the grandchild down with it.
  import subprocess
  child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  open(os.environ['PID_FILE'] + '.child','w').write(str(child.pid))
 print(json.dumps({'type':'stream_event','event':{'type':'content_block_delta','delta':{'type':'text_delta','text':'started'}}}),flush=True)
 time.sleep(60)
settings=json.loads(pathlib.Path(sys.argv[sys.argv.index('--settings')+1]).read_text())
wire=json.loads(settings['env']['CLAUDE_CODE_EXTRA_BODY'])
assert wire['tools'][0]['description'].endswith('TAIL')
assert '--max-turns' in sys.argv and sys.argv[sys.argv.index('--max-turns')+1]=='1'
assert sys.argv[sys.argv.index('--permission-mode')+1]=='dontAsk'
assert sys.argv[sys.argv.index('--tools')+1]==''
assert rows[-1]['type']=='user'
assert 'metadata' not in wire
if os.environ.get('EXPECT_CACHE_TTL'):
 assert os.environ.get('CLAUDE_CODE_PROMPT_CACHE_TTL')==os.environ['EXPECT_CACHE_TTL']
 assert os.environ.get('CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL')==os.environ['EXPECT_CACHE_TTL']
 assert 'FORCE_PROMPT_CACHING_5M' not in os.environ
blocks=[{'type':'thinking','thinking':'private','signature':'signed-test'}, {'type':'text','text':'hello\n'}, {'type':'tool_use','id':'toolu_test','name':os.environ.get('TOOL_NAME','mcp__hermes__probe'),'input':{'value':'x'}}]
for i,n in enumerate(json.loads(os.environ.get('EXTRA_TOOL_NAMES','[]'))):
 blocks.append({'type':'tool_use','id':'toolu_extra_%d'%i,'name':n,'input':{'value':'fixture-argument-%d'%i}})
if len(rows)>1:
 if rows[1]['message']['content'][0]['type']=='thinking':
  assert rows[1]['message']['content']==blocks
 else:
  assert rows[1]['message']['content'][0]['text']=='middleware changed'
 blocks=[{'type':'text','text':'done'}]
for b in blocks:
 if b['type']=='thinking':
  print(json.dumps({'type':'stream_event','event':{'type':'content_block_delta','delta':{'type':'thinking_delta','thinking':b['thinking']}}}),flush=True)
 if b['type']=='text':
  print(json.dumps({'type':'stream_event','event':{'type':'content_block_delta','delta':{'type':'text_delta','text':b['text']}}}),flush=True)
print(json.dumps({'type':'assistant','message':{'role':'assistant','content':blocks,'id':'msg_test','model':'sonnet','stop_reason':'tool_use' if len(blocks)>1 else 'end_turn'}}),flush=True)
print(json.dumps({'type':'stream_event','event':{'type':'message_stop'}}),flush=True)
u={'input_tokens':3,'output_tokens':5,'cache_read_input_tokens':7,'cache_creation_input_tokens':11,'output_tokens_details':{'thinking_tokens':4}}
if 'CACHE_CREATION' in os.environ:
 u['cache_creation']=json.loads(os.environ['CACHE_CREATION'])
print(json.dumps({'type':'result','num_turns':2 if len(blocks)>1 else 1,'subtype':'error_max_turns' if len(blocks)>1 else 'success','is_error':len(blocks)>1,'usage':u,'total_cost_usd':.012345,'modelUsage':{'sonnet':{'costBasis':'list'}}}),flush=True)
sys.exit(1 if len(blocks)>1 else 0)
"""


def _wait_gone(*pids, timeout=15):
    # Reaping is event-driven; the bound only guards a hang and must tolerate a loaded runner.
    import psutil

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE for pid in pids):
            return True
        time.sleep(0.02)
    return False


class Contract(unittest.TestCase):
    def client(self, tmp, **kw):
        import directsdk

        script = Path(tmp) / "native.py"
        script.write_text(FAKE)
        return directsdk.Client(
            command=[sys.executable, str(script)],
            env={"PATH": os.environ["PATH"], "HOME": tmp, **kw},
        )

    def request(self) -> dict:
        return dict(
            model="sonnet",
            messages=[{"role": "user", "content": "go"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "probe",
                        "description": "long " * 600 + "TAIL",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        },
                    },
                }
            ],
        )

    def test_canonical_request_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client(tmp)
            self.assertEqual(client.api_key, "external-process")
            self.assertEqual(client.base_url, "process://claude-subscription-directsdk-experimental")
            for streaming in (False, True):
                req = self.request()
                req["timeout"] = SimpleNamespace(read=10)
                result = client.chat.completions.create(**req, stream=streaming)
                if streaming:
                    chunks = list(result)
                    self.assertEqual(''.join(getattr(c.choices[0].delta, 'reasoning_content', None) or '' for c in chunks), 'private')
                    self.assertEqual(
                        "".join(
                            c.choices[0].delta.content or ""
                            for c in chunks
                            if c.choices
                        ),
                        "hello\n",
                    )
                    final = chunks[-1]
                    msg = {
                        "role": "assistant",
                        "content": "hello",
                        "tool_calls": [
                            {
                                "id": "toolu_test",
                                "type": "function",
                                "function": {
                                    "name": "probe",
                                    "arguments": '{"value":"x"}',
                                },
                            }
                        ],
                        "reasoning_details": final.choices[0].delta.reasoning_details,
                    }
                    self.assertEqual(final.choices[0].finish_reason, "tool_calls")
                else:
                    final = result
                    msg = result.choices[0].message.model_dump()
                    self.assertEqual(msg['reasoning_content'], 'private')
                    self.assertEqual(msg["tool_calls"][0]["function"]["name"], "probe")
                self.assertEqual(final.usage.prompt_tokens, 21)
                self.assertEqual(final.usage.completion_tokens, 5)
                from agent.usage_pricing import normalize_usage
                canonical = normalize_usage(final.usage, api_mode='chat_completions')
                self.assertEqual(canonical.reasoning_tokens, 4)
                self.assertEqual(final.usage.model_dump()['native_cost'], {'total_cost_usd': .012345, 'modelUsage': {'sonnet': {'costBasis': 'list'}}})
                msg["content"] = (msg.get("content") or "").strip()
                req["messages"] += [
                    msg,
                    {
                        "role": "tool",
                        "tool_call_id": "toolu_test",
                        "content": " host\n result",
                    },
                ]
                self.assertEqual(
                    client.chat.completions.create(**req).choices[0].message.content,
                    "done",
                )
                msg["content"] = "middleware changed"
                self.assertEqual(client.chat.completions.create(**req).choices[0].message.content, "done")
            client.close()

    def test_tool_outside_the_inventory_reaches_the_host(self):
        # Hermes errors an unknown tool call back to the model; the transport must not fail the turn.
        with tempfile.TemporaryDirectory() as tmp:
            for native_name, host_name in (("mcp__hermes__ghost", "ghost"), ("mcp__other__ghost", "mcp__other__ghost")):
                client = self.client(tmp, TOOL_NAME=native_name)
                for streaming in (False, True):
                    result = client.chat.completions.create(**self.request(), stream=streaming)
                    if streaming:
                        calls = [tc for c in result if c.choices for tc in (c.choices[0].delta.tool_calls or [])]
                    else:
                        calls = result.choices[0].message.tool_calls
                    self.assertEqual([c.function.name for c in calls], [host_name])
                client.close()

    def test_logged_out_native_raises_the_login_hint(self):
        import directsdk
        from directsdk_setup import LOGGED_OUT_HINT

        with tempfile.TemporaryDirectory() as tmp:
            client = self.client(tmp, NATIVE_ERROR="authentication_failed:Not logged in \u00b7 Please run /login")
            for streaming in (False, True):
                with self.assertRaises(directsdk.ClaudeCodeLoggedOut) as raised:
                    result = client.chat.completions.create(**self.request(), stream=streaming)
                    if streaming:
                        list(result)
                self.assertIn(LOGGED_OUT_HINT, str(raised.exception))
                self.assertIn("Not logged in", str(raised.exception))
            client.close()
            # Any other error native answers itself keeps its own text.
            other = self.client(tmp, NATIVE_ERROR="unknown:API Error: something else")
            with self.assertRaisesRegex(RuntimeError, "^Native API error: API Error: something else$") as raised:
                other.chat.completions.create(**self.request())
            self.assertNotIsInstance(raised.exception, directsdk.ClaudeCodeLoggedOut)
            other.close()

    def test_fail_closed_and_cancellation(self):
        import directsdk

        disabled = json.loads(
            directsdk.request_body(
                {**self.request(), "extra_body": {"reasoning": {"enabled": False}}}
            )[0]
        )
        self.assertEqual(disabled["thinking"], {"type": "disabled"})
        self.assertEqual(disabled["context_management"], {"edits": []})
        # Fable rejects the disable (HTTP 400 "thinking.type.disabled is not supported"), so a
        # caller's disable is omitted rather than sent: thinking stays on, the request survives.
        mandatory = json.loads(
            directsdk.request_body(
                {**self.request(), "model": "fable", "extra_body": {"reasoning": {"enabled": False}}}
            )[0]
        )
        self.assertNotIn("thinking", mandatory)
        self.assertNotIn("context_management", mandatory)
        effort = json.loads(
            directsdk.request_body(
                {**self.request(), "extra_body": {"reasoning": {"effort": "low"}}}
            )[0]
        )
        self.assertEqual(effort["output_config"], {"effort": "low"})
        enabled = {"reasoning": {"enabled": True, "effort": "medium"}}
        adaptive = json.loads(
            directsdk.request_body({**self.request(), "extra_body": enabled})[0]
        )
        self.assertEqual(adaptive["thinking"], {"type": "adaptive"})
        self.assertEqual(adaptive["output_config"], {"effort": "medium"})
        for route in ("haiku", "claude-haiku-4-5", "claude-haiku-4-5-20251001"):
            # Haiku 4.5 answers `adaptive thinking is not supported on this model` with a 400.
            haiku = json.loads(
                directsdk.request_body(
                    {**self.request(), "model": route, "extra_body": enabled}
                )[0]
            )
            self.assertNotIn("thinking", haiku)
            self.assertEqual(haiku["output_config"], {"effort": "medium"})
        off = json.loads(
            directsdk.request_body(
                {
                    **self.request(),
                    "model": "haiku",
                    "extra_body": {"reasoning": {"enabled": False}},
                }
            )[0]
        )
        self.assertEqual(off["thinking"], {"type": "disabled"})
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        }
        fmt = {
            "type": "json_schema",
            "json_schema": {"name": "title", "strict": True, "schema": schema},
        }
        formatted = json.loads(
            directsdk.request_body(
                {**self.request(), "extra_body": {"response_format": fmt}}
            )[0]
        )
        self.assertEqual(
            formatted["output_config"]["format"],
            {"type": "json_schema", "schema": schema},
        )
        routed = directsdk.Client(
            env={"CLAUDE_SUBSCRIPTION_DIRECTSDK_COMMAND": "/native/test"}
        )
        self.assertEqual(routed.command, ["/native/test"])
        from unittest.mock import patch

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-conflicting-key"}):
            with self.assertRaisesRegex(ValueError, "OAuth"):
                directsdk.Client(command="/does/not/exist").create(**self.request())
        client = directsdk.Client(command="/does/not/exist", env={})
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            client.chat.completions.create(**self.request())
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client(tmp)
            for bad in (
                {"extra_body": {"metadata": {}}},
                {"extra_body": []},
                {"temperature": float("nan")},
                {"tool_choice": "required"},
                {"n": 2},
            ):
                with self.assertRaises(ValueError):
                    client.chat.completions.create(**self.request(), **bad)

            async def run():
                result = await client.chat.completions.create(**self.request())
                self.assertEqual(result.choices[0].finish_reason, "tool_calls")

            asyncio.run(run())
            cancel_pidfile = Path(tmp) / "cancel-pid"
            hanging = self.client(tmp, HANG="1", PID_FILE=str(cancel_pidfile))
            stream = hanging.chat.completions.create(**self.request(), stream=True)
            self.assertEqual(next(stream).choices[0].delta.content, "started")
            native_pid, grandchild_pid = int(cancel_pidfile.read_text()), int((Path(str(cancel_pidfile) + ".child")).read_text())
            hanging.cancel()
            with self.assertRaisesRegex(RuntimeError, "cancel"):
                list(stream)
            # cancel() must take the whole tree down, on every OS (Windows: shim -> node grandchild).
            self.assertTrue(_wait_gone(native_pid, grandchild_pid), "cancel() left native or its grandchild running")
            hanging.close()
            # Closing a paused stream must reap without asking for another chunk.
            pidfile = Path(tmp) / "pid"
            hanging = self.client(tmp, HANG="1", PID_FILE=str(pidfile))
            paused = hanging.chat.completions.create(**self.request(), stream=True)
            next(paused)
            pid = int(pidfile.read_text())
            process = paused.request.process
            paused.close()
            self.assertTrue(process.stdout.closed)
            self.assertEqual(len(hanging._requests), 0)
            self.assertTrue(_wait_gone(pid), "Closed paused stream left its native child unreaped")
            hanging.close()
            unstarted = self.client(tmp)
            stream = unstarted.create(**self.request(), stream=True)
            unstarted.close()
            self.assertEqual(len(unstarted._requests), 0)


    def test_tool_schemas_are_normalized_for_the_native_validator(self):
        """Anthropic hard-400s on top-level oneOf/allOf/anyOf and on the null branch of nullable
        unions. The host normalizes both, but only on the api_mode='messages' path, so a single
        Hermes tool carrying a conditional-required hint would fail every request on this
        transport. Nested unions that are not nullable stay as the tool declared them."""
        import directsdk

        req = self.request()
        req["tools"][0]["function"]["parameters"] = {
            "type": "object",
            "properties": {
                "mode": {"type": "string"},
                "proposal": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "ref": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            },
            "required": ["mode"],
            "allOf": [{"if": {"properties": {"mode": {"const": "proposal"}}},
                       "then": {"required": ["proposal"]}}],
        }
        encoded, manifest, _ = directsdk.request_body(req)
        wire = json.loads(encoded)["tools"][0]["input_schema"]
        # The inert MCP manifest and the request body must advertise the same shape.
        for schema in (wire, manifest[0]["inputSchema"]):
            self.assertNotIn("allOf", schema)
            self.assertEqual(schema["required"], ["mode"])
            self.assertEqual(schema["properties"]["proposal"], {"type": "string"})
            self.assertEqual(schema["properties"]["ref"],
                             {"oneOf": [{"type": "string"}, {"type": "integer"}]})
        # A combinator-only schema still reaches the validator as a usable object.
        req["tools"][0]["function"]["parameters"] = {"anyOf": [{"type": "object"}]}
        bare = json.loads(directsdk.request_body(req)[0])["tools"][0]["input_schema"]
        self.assertEqual(bare, {"type": "object", "properties": {}})


CACHE_TTL_KNOB = 'CLAUDE_SUBSCRIPTION_DIRECTSDK_CACHE_TTL'


@pytest.mark.parametrize('policy', [None, '1h', '5m'])
def test_cache_ttl_explicit_env_policy(tmp_path, policy):
    fixture = Contract()
    env = {'EXPECT_CACHE_TTL': policy or '1h', 'FORCE_PROMPT_CACHING_5M': '1',
           'CLAUDE_CODE_PROMPT_CACHE_TTL': '5m', 'CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL': '5m'}
    if policy is not None:
        env[CACHE_TTL_KNOB] = policy
    client = fixture.client(tmp_path, **env)
    try:
        result = client.create(**fixture.request())
        assert result.usage.model_dump()['native_cache_tiers']['requested_ttl'] == (policy or '1h')
    finally:
        client.close()


@pytest.mark.parametrize('policy', [None, '5m'])
def test_cache_ttl_inherited_env_policy(tmp_path, policy):
    fixture = Contract()
    client = fixture.client(tmp_path)
    client.env = None  # Exercise the normal inherited-environment conflict guard too.
    env = {'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'EXPECT_CACHE_TTL': policy or '1h',
           'FORCE_PROMPT_CACHING_5M': '1', 'CLAUDE_CODE_PROMPT_CACHE_TTL': '5m',
           'CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL': '5m'}
    if policy is not None:
        env[CACHE_TTL_KNOB] = policy
    try:
        with patch.dict(os.environ, env, clear=True):
            result = client.create(**fixture.request())
        assert result.usage.model_dump()['native_cache_tiers']['requested_ttl'] == (policy or '1h')
    finally:
        client.close()


@pytest.mark.parametrize('explicit_policy', [None, '1h'])
def test_cache_ttl_explicit_env_does_not_read_process_knob(tmp_path, explicit_policy):
    fixture = Contract()
    env = {'EXPECT_CACHE_TTL': '1h'}
    if explicit_policy:
        env[CACHE_TTL_KNOB] = explicit_policy
    client = fixture.client(tmp_path, **env)
    try:
        with patch.dict(os.environ, {CACHE_TTL_KNOB: '5m'}):
            result = client.create(**fixture.request())
        assert result.usage.model_dump()['native_cache_tiers']['requested_ttl'] == '1h'
    finally:
        client.close()


@pytest.mark.parametrize('invalid', ['', '1H', 'unsafe-value@example.invalid'])
def test_cache_ttl_invalid_warns_once_and_defaults_to_1h(tmp_path, caplog, invalid):
    fixture = Contract()
    client = fixture.client(tmp_path, EXPECT_CACHE_TTL='1h', **{CACHE_TTL_KNOB: invalid})
    try:
        with caplog.at_level('INFO', logger='directsdk'):
            for _ in range(2):
                result = client.create(**fixture.request())
                assert result.usage.model_dump()['native_cache_tiers']['requested_ttl'] == '1h'
        warnings = [r.getMessage() for r in caplog.records if r.levelname == 'WARNING']
        assert warnings == [f'Invalid {CACHE_TTL_KNOB}; using 1h']
        assert 'unsafe-value@example.invalid' not in caplog.text
    finally:
        client.close()


@pytest.mark.parametrize('source', ['delegated', 'cron'])
def test_cache_ttl_query_sources_use_1h(tmp_path, source):
    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
    fixture = Contract()
    client = fixture.client(tmp_path, EXPECT_CACHE_TTL='1h', HERMES_DELEGATED_CHILD_CONTEXT='1')
    context = delegated_child_context if source == 'delegated' else non_dispatcher_owned_context
    try:
        async def run():
            with context():
                result = await client.create(**fixture.request())
            assert result.usage.model_dump()['native_cache_tiers']['requested_ttl'] == '1h'
        asyncio.run(run())
    finally:
        client.close()


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('creation,expected', [
    (None, {'write_5m': None, 'write_1h': None}),
    ({'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 11}, {'write_5m': 0, 'write_1h': 11}),
    ({'ephemeral_5m_input_tokens': 0}, {'write_5m': 0, 'write_1h': None}),
    ({'ephemeral_1h_input_tokens': 0}, {'write_5m': None, 'write_1h': 0}),
    ({'ephemeral_5m_input_tokens': 'unsafe-value@example.invalid', 'ephemeral_1h_input_tokens': -1},
     {'write_5m': None, 'write_1h': None}),
])
def test_cache_tier_telemetry_counts_and_unknown(tmp_path, caplog, streaming, creation, expected):
    fixture = Contract()
    env = {} if creation is None else {'CACHE_CREATION': json.dumps(creation)}
    client = fixture.client(tmp_path, **env)
    try:
        with caplog.at_level('INFO', logger='directsdk'):
            result = client.create(**fixture.request(), stream=streaming)
            final = list(result)[-1] if streaming else result
        usage = final.usage.model_dump()
        assert usage['native_cache_tiers'] == {'requested_ttl': '1h', **expected}
        assert usage['native_usage'].get('cache_creation') == creation
        info = [r.getMessage() for r in caplog.records if r.levelname == 'INFO']
        count = lambda value: 'unknown' if value is None else str(value)
        assert info == [f"DirectSDK cache tiers requested_ttl=1h write_5m={count(expected['write_5m'])} write_1h={count(expected['write_1h'])}"]
        assert not [r for r in caplog.records if r.levelname == 'WARNING']
        assert 'unsafe-value@example.invalid' not in caplog.text
    finally:
        client.close()


@pytest.mark.parametrize('requested', ['1h', '5m'])
def test_cache_tier_mismatch_warns_once_per_client(tmp_path, caplog, requested):
    fixture = Contract()
    creation = {'ephemeral_5m_input_tokens': 3, 'ephemeral_1h_input_tokens': 8}
    env = {CACHE_TTL_KNOB: requested, 'CACHE_CREATION': json.dumps(creation)}
    client = fixture.client(tmp_path, **env)
    try:
        with caplog.at_level('INFO', logger='directsdk'):
            for _ in range(2):
                client.create(**fixture.request())
        messages = [r.getMessage() for r in caplog.records if r.levelname == 'WARNING']
        assert messages == [f'DirectSDK cache tier mismatch requested_ttl={requested} write_5m=3 write_1h=8']
        assert len([r for r in caplog.records if r.levelname == 'INFO']) == 2
    finally:
        client.close()
    # A new client can report its own first mismatch; there is no warning flood per call.
    other = fixture.client(tmp_path, **env)
    try:
        caplog.clear()
        with caplog.at_level('WARNING', logger='directsdk'):
            other.create(**fixture.request())
        assert len(caplog.records) == 1
    finally:
        other.close()


def test_cache_tier_failed_native_call_records_unknown(tmp_path, caplog):
    fixture = Contract()
    client = fixture.client(tmp_path, NATIVE_ERROR='unknown:Public fixture failure')
    try:
        with caplog.at_level('INFO', logger='directsdk'), pytest.raises(RuntimeError, match='Public fixture failure'):
            client.create(**fixture.request())
        assert [r.getMessage() for r in caplog.records if r.levelname == 'INFO'] == [
            'DirectSDK cache tiers requested_ttl=1h write_5m=unknown write_1h=unknown']
    finally:
        client.close()


def test_cache_tier_concurrent_mismatch_warning_is_rate_limited(tmp_path, caplog):
    from concurrent.futures import ThreadPoolExecutor
    fixture = Contract()
    client = fixture.client(tmp_path, CACHE_CREATION=json.dumps({'ephemeral_5m_input_tokens': 11}))
    try:
        with caplog.at_level('INFO', logger='directsdk'), ThreadPoolExecutor(max_workers=4) as pool:
            results = [pool.submit(client.create, **fixture.request()) for _ in range(4)]
            for result in results:
                assert result.result(timeout=10).usage.model_dump()['native_cache_tiers']['write_5m'] == 11
        assert len([r for r in caplog.records if r.levelname == 'INFO']) == 4
        assert [r.getMessage() for r in caplog.records if r.levelname == 'WARNING'] == [
            'DirectSDK cache tier mismatch requested_ttl=1h write_5m=11 write_1h=unknown']
    finally:
        client.close()


@pytest.mark.parametrize('failure', ['version', 'constructor'])
def test_cache_wire_eval_restores_observer_on_setup_failure(tmp_path, monkeypatch, failure):
    import admission
    import importlib.util
    import subprocess
    import threading
    spec = importlib.util.spec_from_file_location('cache_eval_fixture', ROOT / 'evals/directsdk_cache_wire.py')
    eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval)
    original_pin = admission.pin_message_breakpoint
    original_threads = set(threading.enumerate())
    if failure == 'version':
        def fail(*args, **kwargs):
            raise RuntimeError('Public fixture version failure')
        monkeypatch.setattr(subprocess, 'check_output', fail)
    else:
        monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: 'fixture version')
        # The eval imports transport itself; fail that imported client's construction only.
        from importlib.machinery import SourceFileLoader
        original_load = SourceFileLoader.exec_module
        def load(loader, module):
            original_load(loader, module)
            if module.__name__ == 'cache_directsdk':
                def fail(*args, **kwargs):
                    raise RuntimeError('Public fixture constructor failure')
                module.Client = fail
        monkeypatch.setattr(SourceFileLoader, 'exec_module', load)
    with pytest.raises(RuntimeError, match='Public fixture'):
        eval.run(tmp_path / 'not-a-binary', 'sonnet', account_context=True)
    assert admission.pin_message_breakpoint is original_pin
    assert set(threading.enumerate()) == original_threads


@pytest.mark.parametrize('emission', ['info', 'mismatch_warning', 'invalid_warning'])
@pytest.mark.parametrize('native_failure', [False, True])
@pytest.mark.parametrize('streaming', [False, True])
def test_cache_logging_failure_preserves_result_and_cleanup(tmp_path, monkeypatch, emission, native_failure, streaming):
    import directsdk
    import logging

    prefixes = {'info': 'DirectSDK cache tiers', 'mismatch_warning': 'DirectSDK cache tier mismatch',
                'invalid_warning': 'Invalid ' + CACHE_TTL_KNOB}
    attempts = []
    class BrokenHandler(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith(prefixes[emission]):
                attempts.append(record.levelname)
                raise OSError('Public fixture logging sink failure')

    fixture = Contract()
    env = {'EXPECT_CACHE_TTL': '1h'}
    expected_error = 'Native API error: Public fixture native failure'
    if emission == 'mismatch_warning':
        env['CACHE_CREATION'] = json.dumps({'ephemeral_5m_input_tokens': 3, 'ephemeral_1h_input_tokens': 8})
        # Fail after complete native usage is available so the mismatch warning is attempted too.
        if native_failure:
            expected_error = 'Native final text differs from incremental stream'
    elif native_failure:
        env['NATIVE_ERROR'] = 'unknown:Public fixture native failure'
    if emission == 'invalid_warning':
        env[CACHE_TTL_KNOB] = 'invalid'
    client = fixture.client(tmp_path, **env)
    if native_failure and emission == 'mismatch_warning':
        (tmp_path / 'native.py').write_text(FAKE.replace("'text':b['text']", "'text':'Public fixture different text'"))

    gates, processes = [], []
    original_admission, original_spawn = directsdk.Admission, directsdk.Request.spawn
    def gate(*args, **kwargs):
        admission = original_admission(*args, **kwargs)
        gates.append(admission)
        return admission
    def spawn(request, *args, **kwargs):
        process = original_spawn(request, *args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(directsdk, 'Admission', gate)
    monkeypatch.setattr(directsdk.Request, 'spawn', spawn)
    handler = BrokenHandler()
    original_level = directsdk.logger.level
    directsdk.logger.addHandler(handler)
    directsdk.logger.setLevel(logging.INFO)
    try:
        result = error = None
        try:
            result = client.create(**fixture.request(), stream=streaming)
            if streaming:
                result = list(result)[-1]._response
        except Exception as exc:
            error = exc
        # Check lifecycle before any explicit fixture cleanup, including native pipe closure.
        assert len(gates) == 1
        assert all(not admission.thread.is_alive() and admission.server.fileno() == -1 for admission in gates)
        assert not client._requests
        assert len(processes) == 1
        assert all(process.poll() is not None and process.stdin.closed and process.stdout.closed for process in processes)
        assert attempts == ['INFO' if emission == 'info' else 'WARNING']
        if native_failure:
            assert type(error) is RuntimeError
            assert str(error) == expected_error
        else:
            assert error is None
            assert result.choices[0].message.content == 'hello\n'
            assert result.choices[0].message.tool_calls[0].function.name == 'probe'
    finally:
        directsdk.logger.removeHandler(handler)
        directsdk.logger.setLevel(original_level)
        client.close()
        # Clean up resources even on the RED transport, which skips its normal teardown.
        for admission in gates:
            if admission.thread.is_alive() or admission.server.fileno() != -1:
                admission.close()
        for process in processes:
            directsdk.kill_process_tree(process)
            process.wait(timeout=5)
            for pipe in (process.stdin, process.stdout):
                if pipe is not None and not pipe.closed:
                    pipe.close()


OUTSIDE_WARNING = 'DirectSDK native tool call outside the offered inventory: %r; handed to Hermes as %r'
# (name native emitted, name Hermes receives); the request offers only `probe`.
OUTSIDE_NAMES = [
    ('mcp__fastmail__draft_email', 'mcp__fastmail__draft_email'),  # bare MCP name (#39: deferred tool called directly)
    ('Bash', 'Bash'),                                              # a native built-in, never offered (--tools '')
    ('mcp__hermes__ghost', 'ghost'),                               # inert-server prefix, name not in the inventory
    ('probe', 'probe'),                                            # offered name with the prefix dropped (#62)
]


def _tool_calls(result, streaming):
    if streaming:
        chunks = list(result)
        calls = [tc for c in chunks if c.choices for tc in (c.choices[0].delta.tool_calls or [])]
        return chunks[-1]._response, calls
    return result, result.choices[0].message.tool_calls


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('native_name,host_name', OUTSIDE_NAMES)
def test_outside_inventory_call_reaches_hermes_beside_a_valid_parallel_call(tmp_path, caplog, streaming, native_name, host_name):
    fixture = Contract()
    client = fixture.client(tmp_path, EXTRA_TOOL_NAMES=json.dumps([native_name]))
    try:
        with caplog.at_level('INFO', logger='directsdk'):
            response, calls = _tool_calls(client.create(**fixture.request(), stream=streaming), streaming)
        assert response.choices[0].finish_reason == 'tool_calls'
        assert [(c.id, c.function.name) for c in calls] == [('toolu_test', 'probe'), ('toolu_extra_0', host_name)]
        assert json.loads(calls[1].function.arguments) == {'value': 'fixture-argument-0'}
        # The rejected name, once, and nothing from its arguments.
        warnings = [r.getMessage() for r in caplog.records if r.levelname == 'WARNING']
        assert warnings == [OUTSIDE_WARNING % (native_name, host_name)]
        assert 'fixture-argument' not in caplog.text
        # Replay is unchanged: the carrier keeps native's own blocks, and the next round
        # (Hermes' error result for the unknown call) replays them verbatim (FAKE asserts it).
        message = response.choices[0].message.model_dump()
        carried = [b['name'] for m in message['reasoning_details'][0]['messages'] for b in m['content'] if b['type'] == 'tool_use']
        assert carried == ['mcp__hermes__probe', native_name]
        message['content'] = (message['content'] or '').strip()
        request = fixture.request()
        request['messages'] += [message, {'role': 'tool', 'tool_call_id': 'toolu_test', 'content': 'ok'},
                                {'role': 'tool', 'tool_call_id': 'toolu_extra_0',
                                 'content': f"Tool '{host_name}' does not exist. Available tools: probe"}]
        assert client.create(**request).choices[0].message.content == 'done'
    finally:
        client.close()


def test_outside_inventory_warns_once_per_occurrence(tmp_path, caplog):
    fixture = Contract()
    client = fixture.client(tmp_path, EXTRA_TOOL_NAMES=json.dumps(['Bash', 'mcp__hermes__ghost']))
    try:
        with caplog.at_level('WARNING', logger='directsdk'):
            for _ in range(2):
                names = [c.function.name for c in client.create(**fixture.request()).choices[0].message.tool_calls]
                assert names == ['probe', 'Bash', 'ghost']
        assert [r.getMessage() for r in caplog.records] == 2 * [
            OUTSIDE_WARNING % ('Bash', 'Bash'), OUTSIDE_WARNING % ('mcp__hermes__ghost', 'ghost')]
    finally:
        client.close()


def test_offered_names_map_exactly_before_any_prefix_strip(tmp_path, caplog):
    # A host tool may itself be named mcp__hermes__* (an MCP server called "hermes"): an exact
    # inventory match wins, so a dropped outer prefix still names the offered tool.
    fixture = Contract()
    request = fixture.request()
    request['tools'].append({'type': 'function', 'function': {'name': 'mcp__hermes__note', 'parameters': {'type': 'object'}}})
    client = fixture.client(tmp_path, EXTRA_TOOL_NAMES=json.dumps(['mcp__hermes__mcp__hermes__note', 'mcp__hermes__note']))
    try:
        with caplog.at_level('WARNING', logger='directsdk'):
            names = [c.function.name for c in client.create(**request).choices[0].message.tool_calls]
        assert names == ['probe', 'mcp__hermes__note', 'mcp__hermes__note']
        assert [r.getMessage() for r in caplog.records] == [OUTSIDE_WARNING % ('mcp__hermes__note', 'mcp__hermes__note')]
    finally:
        client.close()


@pytest.mark.parametrize('streaming', [False, True])
def test_outside_inventory_logging_failure_cannot_break_the_turn(tmp_path, monkeypatch, streaming):
    import directsdk
    import logging

    attempts = []
    class BrokenHandler(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith('DirectSDK native tool call outside'):
                attempts.append(record.levelname)
                raise OSError('Public fixture logging sink failure')

    processes, original_spawn = [], directsdk.Request.spawn
    def spawn(request, *args, **kwargs):
        process = original_spawn(request, *args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(directsdk.Request, 'spawn', spawn)
    fixture = Contract()
    client = fixture.client(tmp_path, EXTRA_TOOL_NAMES=json.dumps(['Bash']))
    handler, original_level = BrokenHandler(), directsdk.logger.level
    directsdk.logger.addHandler(handler)
    directsdk.logger.setLevel(logging.INFO)
    try:
        response, calls = _tool_calls(client.create(**fixture.request(), stream=streaming), streaming)
        assert attempts == ['WARNING']
        assert [c.function.name for c in calls] == ['probe', 'Bash']
        assert response.choices[0].message.content == 'hello\n'
        assert not client._requests
        assert len(processes) == 1 and processes[0].poll() is not None and processes[0].stdout.closed
    finally:
        directsdk.logger.removeHandler(handler)
        directsdk.logger.setLevel(original_level)
        client.close()


# A native that answers the first query with the tool calls in LOOP_TOOLS, then "done";
# it records every stdin frame so the test can read what the second request replayed.
LOOP_NATIVE = r"""
import json, os, sys
rows=[]
for line in sys.stdin:
 r=json.loads(line); rows.append(r)
 if r.get('shouldQuery') is False:
  print(json.dumps({'type':'result','num_turns':0,'is_error':False}),flush=True)
with open(os.environ['ROWS_FILE'],'a') as f:
 f.write(json.dumps(rows)+'\n')
if len(rows)==1:
 blocks=[{'type':'tool_use','id':'toolu_loop_%d'%i,'name':n,'input':{'query':'fixture-argument'}} for i,n in enumerate(json.loads(os.environ['LOOP_TOOLS']))]
else:
 blocks=[{'type':'text','text':'done'}]
 print(json.dumps({'type':'stream_event','event':{'type':'content_block_delta','delta':{'type':'text_delta','text':'done'}}}),flush=True)
tool=blocks[0]['type']=='tool_use'
print(json.dumps({'type':'assistant','message':{'role':'assistant','content':blocks,'id':'msg_loop','model':'sonnet','stop_reason':'tool_use' if tool else 'end_turn'}}),flush=True)
print(json.dumps({'type':'stream_event','event':{'type':'message_stop'}}),flush=True)
print(json.dumps({'type':'result','num_turns':1,'subtype':'error_max_turns' if tool else 'success','is_error':tool,'usage':{'input_tokens':1,'output_tokens':1}}),flush=True)
sys.exit(1 if tool else 0)
"""


@pytest.mark.parametrize('with_valid_call', [True, False])
def test_hermes_loop_errors_unoffered_calls_and_runs_only_offered_tools(tmp_path, monkeypatch, with_valid_call):
    """The real Hermes turn loop on this transport. Tool Search defers `todo_list`, so it is a
    registered Hermes tool that this request did not offer: a direct call to it, a native
    built-in and a foreign MCP name must each get Hermes' unknown-tool result and never run,
    while an offered call in the same batch does run and the turn completes."""
    import directsdk
    import model_tools
    from run_agent import AIAgent

    home = tmp_path / 'hermes-home'
    home.mkdir()
    (home / 'config.yaml').write_text('tools:\n  tool_search:\n    enabled: "on"\n    defer: [todo_list]\n', encoding='utf-8')
    monkeypatch.setenv('HERMES_HOME', str(home))
    unoffered = ['Bash', 'mcp__hermes__todo_list', 'mcp__fastmail__draft_email']
    native = (['mcp__hermes__tool_search'] if with_valid_call else []) + unoffered
    script, rows_file = tmp_path / 'loop_native.py', tmp_path / 'rows.jsonl'
    script.write_text(LOOP_NATIVE)
    client = directsdk.Client(command=[sys.executable, str(script)], env={
        'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'LOOP_TOOLS': json.dumps(native), 'ROWS_FILE': str(rows_file)})
    dispatched = []
    def dispatch(name, *args, **kwargs):
        dispatched.append(name)  # Record only: no Hermes tool really runs in this test.
        return json.dumps({'ok': True})
    monkeypatch.setattr(model_tools, 'handle_function_call', dispatch)
    agent = AIAgent(base_url='http://127.0.0.1:9/v1', api_key='fixture', model='sonnet', enabled_toolsets=['todo'],
                    quiet_mode=True, skip_context_files=True, skip_memory=True, skip_background_review=True,
                    checkpoints_enabled=False, load_soul_identity=False, max_iterations=4)
    try:
        assert 'tool_search' in agent.valid_tool_names and 'todo_list' not in agent.valid_tool_names
        from tools.registry import registry
        assert registry.get_entry('todo_list') is not None  # registered, just not offered
        monkeypatch.setattr(agent, '_create_request_openai_client', lambda **_: client)
        monkeypatch.setattr(agent, '_close_request_openai_client', lambda *_, **__: None)
        result = agent.run_conversation('go')
        assert result['completed'] is True and result['final_response'] == 'done'
        assert dispatched == (['tool_search'] if with_valid_call else [])
        results = {m['tool_call_id']: m['content'] for m in result['messages'] if m.get('role') == 'tool'}
        offset = int(with_valid_call)
        for i, name in enumerate(['Bash', 'todo_list', 'mcp__fastmail__draft_email']):
            assert results[f'toolu_loop_{i + offset}'].startswith(f"Tool '{name}' does not exist. Available tools: ")
        # The model sees each error in the next request, so it can correct itself.
        replayed = json.loads(rows_file.read_text().splitlines()[-1])
        sent = {b['tool_use_id']: b['content'] for r in replayed if r['type'] == 'user'
                for b in r['message']['content'] if b.get('type') == 'tool_result'}
        assert sent == results
    finally:
        agent.close()
        client.close()


if __name__ == "__main__":
    unittest.main()

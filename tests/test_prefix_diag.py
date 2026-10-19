"""#17 diagnostics: opt-in, content-free, never changes a request."""
import json
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import directsdk
import prefix_diag

SECRET = 'TOPSECRET-user-text-9f3a'
EMAIL = 'someone@example.invalid'

NATIVE = r'''
import json, os, sys, urllib.request
frames = []
for line in sys.stdin:
    frame = json.loads(line)
    frames.append(frame)
    if frame.get('type') != 'user':
        continue
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type':'result','num_turns':0}), flush=True)
        continue
    break
messages = [f['message'] for f in frames]
body = {'model':'m','system':[{'type':'text','text':'sys'}],'tools':[{'name':'t','input_schema':{}}],
        'messages':messages}
messages[-1]['content'][-1]['cache_control'] = {'type':'ephemeral','ttl':'1h'}
data = json.dumps(body).encode()
urllib.request.urlopen(urllib.request.Request(os.environ['ANTHROPIC_BASE_URL']+'/v1/messages', data=data,
                       headers={'Content-Type':'application/json'}), timeout=5).read()
print(json.dumps({'type':'assistant','message':{'id':'a','role':'assistant','content':[{'type':'text','text':'OK'}]}}))
print(json.dumps({'type':'stream_event','event':{'type':'message_stop'}}))
print(json.dumps({'type':'result','subtype':'success','usage':{'input_tokens':0,'output_tokens':0}}))
'''

USAGE = {'input_tokens': 3, 'output_tokens': 1, 'cache_read_input_tokens': 5, 'cache_creation_input_tokens': 7}


@pytest.fixture
def peer():
    bodies = []

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            bodies.append(self.rfile.read(int(self.headers['Content-Length'])))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            events = [
                {'type': 'message_start', 'message': {'id': 'a', 'role': 'assistant', 'model': 'm', 'content': [], 'usage': USAGE}},
                {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'OK'}},
                {'type': 'content_block_stop', 'index': 0},
                {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': USAGE},
                {'type': 'message_stop'},
            ]
            self.wfile.write(''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, bodies
    server.shutdown()
    thread.join()
    server.server_close()


def run(tmp_path, port, env_extra, messages):
    native = tmp_path / 'native.py'
    native.write_text(NATIVE)
    env = {'PATH': os.defpath, 'HOME': str(tmp_path), 'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{port}', **env_extra}
    client = directsdk.Client(command=[sys.executable, str(native)], env=env)
    try:
        return client.create(model='sonnet', messages=messages)
    finally:
        client.close()


MESSAGES = [
    {'role': 'system', 'content': 'system ' + SECRET},
    {'role': 'user', 'content': 'first ' + SECRET + ' ' + EMAIL},
    {'role': 'assistant', 'content': 'reply ' + SECRET},
    {'role': 'user', 'content': 'second ' + SECRET},
]


def test_enabled_records_hashes_only(tmp_path, peer, monkeypatch):
    server, bodies = peer
    out = tmp_path / 'diag.jsonl'
    monkeypatch.setenv(prefix_diag.ENV, str(out))
    result = run(tmp_path, server.server_port, {}, MESSAGES)
    assert result.choices[0].message.content == 'OK'
    raw = out.read_text()
    assert SECRET not in raw and EMAIL not in raw and 'reply' not in raw
    record = json.loads(raw.splitlines()[0])
    assert record['v'] == 1 and record['seq'] == 1 and record['error'] is None
    assert [row[0] for row in record['host']] == ['system', 'user', 'assistant', 'user']
    assert [row[0] for row in record['frames']] == ['system', 'user', 'assistant', 'user']
    wire = record['wire']['native']
    assert [m[0] for m in wire['messages']] == ['user', 'assistant', 'user']
    assert wire['marks'] == [['m', 2, 0, '{"ttl":"1h","type":"ephemeral"}']]
    assert record['usage'] == {k: USAGE[k] for k in record['usage']}
    # The same block hashes identically at frame and wire layers (cache_control excluded).
    assert record['frames'][3][1][0][1] == wire['messages'][2][1][0][1]


def test_stable_history_hashes_repeat_across_requests(tmp_path, peer, monkeypatch):
    server, _ = peer
    out = tmp_path / 'diag.jsonl'
    monkeypatch.setenv(prefix_diag.ENV, str(out))
    run(tmp_path, server.server_port, {}, MESSAGES)
    run(tmp_path, server.server_port, {}, MESSAGES + [
        {'role': 'assistant', 'content': 'again'}, {'role': 'user', 'content': 'third'}])
    first, second = (json.loads(line) for line in out.read_text().splitlines())
    a, b = first['wire']['native']['messages'], second['wire']['native']['messages']
    assert [m[1] for m in a[:2]] == [m[1] for m in b[:2]]


def test_disabled_writes_nothing_and_forwards_identically(tmp_path, peer, monkeypatch):
    server, bodies = peer
    monkeypatch.delenv(prefix_diag.ENV, raising=False)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    run(tmp_path, server.server_port, {}, MESSAGES)
    monkeypatch.setenv(prefix_diag.ENV, str(tmp_path / 'diag.jsonl'))
    run(tmp_path, server.server_port, {}, MESSAGES)
    assert not (tmp_path / 'home').exists()
    assert bodies[0] == bodies[1]


def test_default_path_enabled_only_when_file_exists(tmp_path, monkeypatch):
    monkeypatch.delenv(prefix_diag.ENV, raising=False)
    home = tmp_path / 'home'
    monkeypatch.setenv('HERMES_HOME', str(home))
    assert prefix_diag.target() is None
    (home / 'logs').mkdir(parents=True)
    (home / 'logs' / prefix_diag.DEFAULT_NAME).touch()
    assert prefix_diag.target() == home / 'logs' / prefix_diag.DEFAULT_NAME
    monkeypatch.setenv(prefix_diag.ENV, 'relative.jsonl')
    assert prefix_diag.target() is None


def test_broken_sink_never_changes_the_result(tmp_path, peer, monkeypatch):
    server, _ = peer
    sink = tmp_path / 'is-a-directory'
    sink.mkdir()
    monkeypatch.setenv(prefix_diag.ENV, str(sink))
    result = run(tmp_path, server.server_port, {}, MESSAGES)
    assert result.choices[0].message.content == 'OK'


def test_failed_request_is_recorded_with_error_type(tmp_path, monkeypatch):
    out = tmp_path / 'diag.jsonl'
    monkeypatch.setenv(prefix_diag.ENV, str(out))
    native = tmp_path / 'native.py'
    native.write_text('import sys\nsys.exit(3)\n')
    client = directsdk.Client(command=[sys.executable, str(native)],
                              env={'PATH': os.defpath, 'HOME': str(tmp_path)})
    with pytest.raises(Exception):
        client.create(model='sonnet', messages=MESSAGES)
    client.close()
    record = json.loads(out.read_text().splitlines()[0])
    assert record['error'] and record['wire'] is None and SECRET not in out.read_text()


def test_size_cap_stops_writing(tmp_path, monkeypatch):
    out = tmp_path / 'diag.jsonl'
    monkeypatch.setenv(prefix_diag.ENV, str(out))
    monkeypatch.setattr(prefix_diag, 'MAX_BYTES', 5)  # the first 8-byte line crosses it
    prefix_diag.write({'a': 1})
    prefix_diag.write({'b': 2})
    assert len(out.read_text().splitlines()) == 1

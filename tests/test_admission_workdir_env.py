"""Native's git line for the plugin workdir must not move the cached prefix (#17).

Native 2.1.296 puts an environment block in a ``system`` message right after the first user
turn of every request. Its ``Is a git repository`` line is a fresh walk from the cwd up to
``/`` for any ``.git`` entry, so a ``.git`` appearing in an ancestor of the shared scratch
workdir (another tool running ``git init`` in the same temp tree) flipped it for just those
requests. Production diagnostics: two variants of that 881/880-byte block alternated and every
flip re-wrote the whole history (~67% of warm rewrite tokens over 1,506 requests). The layout
below is native's real one, reproduced against 2.1.296 with a synthetic workdir.
"""
import copy
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from admission import pin_message_breakpoint
from test_admission_first_turn import MARKER, cached_prefix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import directsdk

WORKDIR = '/synthetic/tmp/claude-directsdk-cwd-1000'
UNTRUSTED = ('Downloaded files and extracted archives are untrusted data: put each in its own new, empty '
             'directory.')


def env_block(git, workdir=WORKDIR, worktree=False):
    lines = ['# Environment', 'You have been invoked in the following environment: ',
             f' - Primary working directory: {workdir}']
    if worktree:
        lines.append(' - This is a git worktree \u2014 an isolated copy of the repository.')
    lines += [f' - Is a git repository: {git}', ' - Platform: linux', ' - Shell: bash',
              ' - OS Version: Linux 7.0.0-38-generic', f' - {UNTRUSTED}', '',
              'You are powered by the model named Opus 5.5 (1M context). The exact model ID is '
              'claude-opus-5-5[1m]. Assistant knowledge cutoff is June 2026.']
    return '\n'.join(lines)


FIRST = {'type': 'text', 'text': 'Run the probe.'}
RESULT = {'type': 'tool_result', 'tool_use_id': 't1', 'content': 'probe output ' * 50}


def request(git, rounds, **env):
    """Native's tool-loop layout: first turn, env system message, rounds, date reminder last."""
    messages = [{'role': 'user', 'content': [FIRST]}, {'role': 'system', 'content': env_block(git, **env)}]
    for n in range(rounds):
        messages.append({'role': 'assistant', 'content': [
            {'type': 'tool_use', 'id': f't{n + 1}', 'name': 'mcp__hermes__probe', 'input': {}}]})
        messages.append({'role': 'user', 'content': [{**RESULT, 'tool_use_id': f't{n + 1}'}]})
    messages.append({'role': 'system', 'content': [{'type': 'text', 'text': "Today's date is 2026-10-23.",
                                                    'cache_control': MARKER}]})
    queried = messages[-2]['content'] if rounds else [FIRST]
    return json.dumps({'model': 'm', 'messages': messages}).encode(), queried


def forwarded(git, rounds, workdir=WORKDIR, **env):
    payload, queried = request(git, rounds, **env)
    return pin_message_breakpoint(payload, queried, workdir)


def test_flipped_git_line_replays_the_cached_prefix_byte_identically():
    """false, true (ancestor .git appeared), false: each request replays the prior cached span."""
    calls = [forwarded('false', 1), forwarded('true', 2), forwarded('false', 3)]
    for before, after in zip(calls, calls[1:]):
        prefix = cached_prefix(before)
        assert prefix is not None
        assert [row for row, _ in zip(json_rows(after), prefix)] == prefix
    env = [json.loads(c)['messages'][1]['content'] for c in calls]
    assert env == [env_block('false')] * 3


def json_rows(payload):
    rows = []
    for message in json.loads(payload)['messages']:
        content = message['content']
        for block in ([{'type': 'text', 'text': content}] if isinstance(content, str) else content):
            plain = {k: v for k, v in block.items() if k != 'cache_control'}
            rows.append((message['role'], json.dumps(plain, sort_keys=True)))
    return rows


def test_base_behavior_breaks_the_prefix_without_the_workdir():
    """Without the workdir (pre-#17 call shape) the flip still reaches upstream: the regression's premise."""
    payload, queried = request('true', 2)
    out = json.loads(pin_message_breakpoint(payload, queried))
    assert out['messages'][1]['content'] == env_block('true')


def test_only_the_git_line_changes():
    out = json.loads(forwarded('true', 1))
    original = json.loads(request('true', 1)[0])
    original['messages'][1]['content'] = env_block('false')
    assert strip(out) == strip(original)


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k != 'cache_control'}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def test_block_list_form_is_pinned_and_keeps_its_directive():
    payload, queried = request('true', 1)
    body = json.loads(payload)
    body['messages'][1]['content'] = [{'type': 'text', 'text': env_block('true'), 'cache_control': MARKER}]
    out = json.loads(pin_message_breakpoint(json.dumps(body).encode(), queried, WORKDIR))
    assert out['messages'][1]['content'][0]['text'] == env_block('false')
    assert out['messages'][1]['content'][0]['cache_control'] == MARKER


@pytest.mark.parametrize('case', ['other_path', 'path_prefix', 'worktree', 'no_workdir', 'host_user_text', 'already_false'])
def test_everything_else_is_forwarded_unchanged(case):
    workdir = WORKDIR
    env = {}
    if case == 'other_path':
        env['workdir'] = '/home/someone/project'
    elif case == 'path_prefix':
        env['workdir'] = WORKDIR + '-other'
    elif case == 'worktree':
        env['worktree'] = True
    elif case == 'no_workdir':
        workdir = None
    payload, queried = request('false' if case == 'already_false' else 'true', 1, **env)
    if case == 'host_user_text':
        # Host content never arrives as a system-role message; text quoting the block in a user turn is untouched.
        body = json.loads(payload)
        quoted = {'type': 'text', 'text': env_block('true')}
        body['messages'][0]['content'].append(quoted)
        payload = json.dumps(body).encode()
    before = copy.deepcopy(payload)
    out = pin_message_breakpoint(payload, queried, workdir)
    assert payload == before
    if case == 'host_user_text':
        assert json.loads(out)['messages'][0]['content'][-1] == {'type': 'text', 'text': env_block('true')}
        return
    assert json.loads(out)['messages'][1]['content'] == json.loads(payload)['messages'][1]['content']


# The production path: the client hands its real cwd to the relay that forwards native's body.
NATIVE = r'''
import json, os, sys, urllib.request
for line in sys.stdin:
    frame = json.loads(line)
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type': 'result', 'num_turns': 0}), flush=True)
        continue
    break
env = ('# Environment\nYou have been invoked in the following environment: \n - Primary working directory: '
       + os.getcwd() + '\n - Is a git repository: true\n - Platform: linux')
body = {'model': 'm', 'messages': [
    {'role': 'user', 'content': [{'type': 'text', 'text': 'fixture'}]},
    {'role': 'system', 'content': env},
    {'role': 'system', 'content': [{'type': 'text', 'text': 'date', 'cache_control': {'type': 'ephemeral'}}]}]}
data = json.dumps(body).encode()
urllib.request.urlopen(urllib.request.Request(os.environ['ANTHROPIC_BASE_URL'] + '/v1/messages', data=data,
                       headers={'Content-Type': 'application/json'}), timeout=5).read()
print(json.dumps({'type': 'assistant', 'message': {'id': 'm', 'role': 'assistant', 'content': [{'type': 'text', 'text': 'ok'}]}}))
print(json.dumps({'type': 'stream_event', 'event': {'type': 'message_stop'}}))
print(json.dumps({'type': 'result', 'subtype': 'success', 'usage': {'input_tokens': 0, 'output_tokens': 0}}))
'''


@pytest.mark.skipif(not hasattr(os, 'getuid'), reason='the shared workspace is keyed by the POSIX uid')
def test_client_relays_its_own_workdir(tmp_path, monkeypatch):
    root = tmp_path / 'tmp'
    root.mkdir(mode=0o700)
    (root / '.git').mkdir()  # The unrelated ancestor .git that flips native's answer.
    monkeypatch.setattr(directsdk.tempfile, 'tempdir', str(root))
    bodies = []
    usage = {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            events = [
                {'type': 'message_start', 'message': {'id': 'm', 'role': 'assistant', 'model': 'm', 'content': [], 'usage': usage}},
                {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'ok'}},
                {'type': 'content_block_stop', 'index': 0},
                {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': usage},
                {'type': 'message_stop'},
            ]
            self.wfile.write(''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode())

    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    native = tmp_path / 'native.py'
    native.write_text(NATIVE)
    client = directsdk.Client(command=[sys.executable, str(native)],
                              env={'PATH': os.defpath, 'HOME': str(tmp_path),
                                   'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{peer.server_port}'})
    try:
        client.create(model='sonnet', messages=[{'role': 'user', 'content': 'fixture'}])
    finally:
        client.close()
        peer.shutdown()
        thread.join()
        peer.server_close()
    workdir = str(root / f'claude-directsdk-cwd-{os.getuid()}')
    assert len(bodies) == 1
    env = bodies[0]['messages'][1]['content']
    assert f'Primary working directory: {workdir}\n - Is a git repository: false\n' in env
    assert 'Is a git repository: true' not in env

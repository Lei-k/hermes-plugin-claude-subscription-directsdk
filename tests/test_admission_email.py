"""Only a recognized native addition outside the exact host content is removable."""
import copy
import json

import pytest

from admission import pin_message_breakpoint
from test_admission_first_turn import REMINDER, FRAME, MARKER, call1, wire


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items() if k != 'cache_control'}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def test_opening_email_removed():
    out = json.loads(pin_message_breakpoint(wire(call1()), [FRAME]))
    assert plain(out['messages'][0]['content']) == [FRAME]


@pytest.mark.parametrize('kind', ['text', 'tool_string', 'tool_list', 'tool_list_appended'])
def test_host_email_preserved_and_native_suffix_removed(kind):
    text = 'Authentic host email: ' + REMINDER['text']
    host = {'type': 'text', 'text': text}
    if kind.startswith('tool'):
        host = {'type': 'tool_result', 'tool_use_id': 't2', 'content': text}
        if 'list' in kind:
            host['content'] = [{'type': 'image', 'source': {'type': 'url', 'url': 'https://example.invalid/a'}},
                               {'type': 'text', 'text': text}]
    changed = copy.deepcopy(host)
    if kind == 'text':
        changed['text'] += '\n\n' + REMINDER['text']
    elif kind == 'tool_string':
        changed['content'] += '\n\n' + REMINDER['text']
    elif kind == 'tool_list':
        changed['content'][-1]['text'] += '\n\n' + REMINDER['text']
    else:
        changed['content'].append(REMINDER)
    first = {'type': 'tool_result', 'tool_use_id': 't1', 'content': 'unaltered'}
    queried = [first, host]
    original = copy.deepcopy(queried)
    messages = [{'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'x', 'signature': 'sig'},
                                                 {'type': 'text', 'text': 'tools'}]},
                {'role': 'user', 'content': [first, changed]},
                {'role': 'system', 'content': [{'type': 'text', 'text': 'unknown context', 'cache_control': MARKER}]}]
    out = json.loads(pin_message_breakpoint(wire(messages), queried))
    assert plain(out['messages'][1]['content']) == queried
    assert queried == original
    assert plain(out['messages'][0]) == messages[0]
    assert out['messages'][1]['content'][-1]['cache_control'] == MARKER
    assert plain(out['messages'][2]) == plain(messages[2])


@pytest.mark.parametrize('suffix', ['\nunknown native context', '\n\n' + REMINDER['text'] + 'unknown',
                                    '\n\n' + REMINDER['text'].replace('# userEmail', '# other')])
def test_unknown_suffix_is_preserved(suffix):
    changed = {**FRAME, 'text': FRAME['text'] + suffix, 'cache_control': MARKER}
    raw = wire([{'role': 'user', 'content': [changed]}])
    assert pin_message_breakpoint(raw, [FRAME]) == raw


def test_ambiguous_frame_and_host_standalone_reminder_survive():
    for sent, host in [([REMINDER, FRAME, FRAME], [FRAME]), ([REMINDER, FRAME], [REMINDER, FRAME])]:
        out = json.loads(pin_message_breakpoint(wire([{'role': 'user', 'content': sent}]), host))
        assert out['messages'][0]['content'] == sent


def test_relay_sanitizes_before_forwarding_and_preserves_identity():
    import http.client
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from admission import Admission

    calls = []
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            calls.append((dict(self.headers), json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'fixture error')

    with ThreadingHTTPServer(('127.0.0.1', 0), Peer) as peer:
        thread = threading.Thread(target=peer.serve_forever, daemon=True)
        thread.start()
        gate = Admission(f'http://127.0.0.1:{peer.server_port}', 5, queried=[FRAME])
        try:
            for _ in range(2):
                conn = http.client.HTTPConnection('127.0.0.1', gate.server.server_port, timeout=5)
                conn.request('POST', gate.prefix + '/v1/messages', wire(call1()),
                             {'Authorization': 'Bearer public-fixture', 'X-Native-Identity': 'fixture'})
                assert conn.getresponse().read()
                conn.close()
            assert len(calls) == 1 and gate.denied == 1
            assert calls[0][0]['Authorization'] == 'Bearer public-fixture'
            assert calls[0][0]['X-Native-Identity'] == 'fixture'
            assert plain(calls[0][1]['messages'][0]['content']) == [FRAME]
            assert gate.error_text() == 'fixture error'
        finally:
            gate.close()
            peer.shutdown()
            thread.join()


NATIVE_REMINDER = {
    'type': 'text',
    'text': "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n"
            "# userEmail\nThe user's email address is public-fixture@example.invalid. "
            "Use it only to identify the user, such as for authorship, attribution, or filtering their own work. "
            "Never send it to an unrelated service, such as in a request header, URL, or payload, "
            "unless the user explicitly asks.\n\n"
            "IMPORTANT: this context may or may not be relevant to your tasks. You should not respond "
            "to this context unless it is highly relevant to your task.\n</system-reminder>\n",
}


def test_native_ordinary_followup_removes_prepended_email_but_keeps_date():
    """Claude 2.1.283 prepends the reminder again after an ordinary assistant answer."""
    host = {'type': 'text', 'text': 'Public followup 2'}
    date = {'role': 'system', 'content': [{'type': 'text',
            'text': "<system-reminder>\nToday's date is 2026-10-03.\n</system-reminder>",
            'cache_control': MARKER}]}
    history = [{'role': 'user', 'content': [NATIVE_REMINDER]},
               {'role': 'assistant', 'content': [{'type': 'text', 'text': 'PUBLIC SYNTHETIC ANSWER'}]}]
    messages = history + [{'role': 'user', 'content': [NATIVE_REMINDER, host]}, date]
    original = copy.deepcopy(messages)
    out = json.loads(pin_message_breakpoint(wire(messages), [host]))
    assert plain(out['messages'][-2]['content']) == [host]
    assert out['messages'][:-2] == history  # authentic historical identical reminder survives
    assert plain(out['messages'][-1]) == plain(date)
    assert out['messages'][-2]['content'][0]['cache_control'] == MARKER
    assert messages == original


def test_reminder_containing_unknown_or_additional_date_context_is_preserved():
    # Without a proven template boundary, keep the whole block, including its nonemail context.
    combined = {**NATIVE_REMINDER, 'text': NATIVE_REMINDER['text'].replace(
        '\n\nIMPORTANT:', "\n# currentDate\nToday's date is 2026-10-03.\n\nIMPORTANT:")}
    raw = wire([{'role': 'user', 'content': [combined, FRAME]}])
    assert pin_message_breakpoint(raw, [FRAME]) == raw


@pytest.mark.parametrize('marker', [{'type': 'ephemeral'}, {'type': 'ephemeral', 'ttl': '1h'}])
def test_normalization_preserves_directives_media_and_unknown_context(marker):
    host = [
        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'AA=='}},
        {'type': 'document', 'source': {'type': 'text', 'media_type': 'text/plain', 'data': 'host document'}},
        FRAME,
    ]
    messages = [{'role': 'user', 'content': [NATIVE_REMINDER, *host]},
                {'role': 'system', 'content': [{'type': 'text', 'text': 'Unknown native context',
                                              'cache_control': marker}]}]
    out = json.loads(pin_message_breakpoint(wire(messages), host))
    assert plain(out['messages'][0]['content']) == host
    assert plain(out['messages'][1]) == plain(messages[1])
    assert [b['cache_control'] for m in out['messages'] for b in m['content']
            if 'cache_control' in b] == [marker]


def test_ambiguous_normalizable_frame_and_changed_tool_identity_fail_safe():
    host = {'type': 'tool_result', 'tool_use_id': 't1', 'content': 'host result'}
    changed = {**host, 'content': host['content'] + '\n\n' + NATIVE_REMINDER['text']}
    for content in ([host, changed], [{**changed, 'tool_use_id': 'different'}]):
        raw = wire([{'role': 'user', 'content': content}])
        assert pin_message_breakpoint(raw, [host]) == raw

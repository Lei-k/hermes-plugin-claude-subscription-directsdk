"""Request-scoped native HTTP admission; credentials are forwarded, never persisted."""
import codecs
import copy
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import re
import secrets
import socket
import ssl
import threading
from urllib.parse import urlsplit


UNCACHEABLE = ('thinking', 'redacted_thinking')


def _plain(block):
    return {k: v for k, v in block.items() if k != 'cache_control'} if isinstance(block, dict) else block


# Deliberately recognize one native template, not arbitrary reminders or email text.
_ACCOUNT_REMINDER = re.compile(
    r"<system-reminder>\nAs you answer the user's questions, you can use the "
    r"following context:\n# userEmail\nThe user's email address is "
    r"[^\s<>@]+@[^\s<>@]+\."
    r"(?: Use it only to identify the user, such as for authorship, attribution, or filtering "
    r"their own work\. Never send it to an unrelated service, such as in a request header, "
    r"URL, or payload, unless the user explicitly asks\.)?"
    r"(?:\n\nIMPORTANT: this context may or may not be relevant to your tasks\. You should "
    r"not respond to this context unless it is highly relevant to your task\.)?"
    r"\n</system-reminder>\n?")


def _account_block(block):
    return (isinstance(block, dict) and set(block) == {'type', 'text'}
            and block['type'] == 'text' and isinstance(block['text'], str)
            and _ACCOUNT_REMINDER.fullmatch(block['text']) is not None)


def _without_account(sent, host):
    """Return a reconciled block, or None when anything beyond the template differs."""
    if _plain(sent) == _plain(host):
        return copy.deepcopy(sent)
    if not isinstance(sent, dict) or not isinstance(host, dict):
        return None
    kind = host.get('type')
    field = 'text' if kind == 'text' else 'content' if kind == 'tool_result' else None
    if field is None or field not in sent or field not in host:
        return None
    left, right = _plain(sent), _plain(host)
    actual, expected = left.pop(field), right.pop(field)
    if left != right:
        return None
    if isinstance(actual, str) and isinstance(expected, str):
        if not actual.startswith(expected):
            return None
        suffix = actual[len(expected):]
        if not any(suffix.startswith(sep) and _ACCOUNT_REMINDER.fullmatch(suffix[len(sep):])
                   for sep in ('\n\n', '\n')):
            return None
        restored = expected
    elif kind == 'tool_result' and isinstance(actual, list) and isinstance(expected, list):
        if len(actual) not in (len(expected), len(expected) + 1):
            return None
        restored = [_without_account(a, b) for a, b in zip(actual, expected)]
        if any(b is None for b in restored):
            return None
        if len(actual) > len(expected) and not _account_block(actual[-1]):
            return None
    else:
        return None
    result = copy.deepcopy(sent)
    result[field] = restored
    return result


def _normalize_account(messages, queried):
    """Anchor a complete unique host frame; preserve unknown surrounding native context."""
    last = max((i for i, m in enumerate(messages) if m.get('role') == 'assistant'), default=-1)
    newest = messages[last + 1] if last + 1 < len(messages) else {}
    content = newest.get('content')
    if newest.get('role') != 'user' or not isinstance(content, list):
        return False
    matches = []
    for start in range(len(content) - len(queried) + 1):
        restored = [_without_account(a, b) for a, b in zip(content[start:], queried)]
        if all(b is not None for b in restored):
            matches.append((start, restored))
    if len(matches) != 1:
        return False
    start, restored = matches[0]
    before, after = content[:start], content[start + len(queried):]
    # Only a standalone reminder immediately before the queried host frame is recognized.
    # Marked/extended reminder blocks fail safe: never discard their directives or metadata.
    if len(before) == 1 and _account_block(before[0]):
        before = []
    if len(after) == 1 and _account_block(after[0]):
        after = []
    normalized = before + restored + after
    if normalized == content:
        return False
    newest['content'] = normalized
    return True


def _frame_first(messages, queried):
    """Put Hermes' first user turn ahead of the context native prepends to it (#77).

    On the opening request native puts its per-request context block (the account-email
    reminder) *before* the frame Hermes queried; on every later request that turn is replayed
    without it. Nothing of the first turn therefore recurs, and call #2 re-writes the whole of
    it, which is most of a cron run whose first turn carries a source pack. On later turns
    native already appends its context after the host content. Moving the prepended text
    blocks behind the frame uses that same order for the first turn: no block is added,
    dropped or edited, and the frame becomes a prefix the next request replays byte-identically.

    Only the conversation's first turn is touched (no assistant message yet), only when the
    queried frame occurs exactly once, and only when everything before it is plain text.
    Returns whether the order changed."""
    if any(m.get('role') == 'assistant' for m in messages):
        return False
    first = next((m for m in messages if m.get('role') == 'user'), {})
    content = first.get('content')
    if not isinstance(content, list) or not queried or len(content) <= len(queried):
        return False
    want = [_plain(b) for b in queried]
    starts = [k for k in range(len(content) - len(queried) + 1)
              if [_plain(b) for b in content[k:k + len(queried)]] == want]
    if len(starts) != 1 or starts[0] == 0:
        return False
    k = starts[0]
    if not all(isinstance(b, dict) and b.get('type') == 'text' for b in content[:k]):
        return False
    first['content'] = content[k:k + len(queried)] + content[:k] + content[k + len(queried):]
    return True


def pin_message_breakpoint(payload, queried):
    """Keep the single message ``cache_control`` on content the next request replays unchanged.

    Native attaches per-request context (today's date, the account-email reminder, whatever a
    later CLI adds) to the turn it answers and puts the message breakpoint on or after it. The
    next request replays that turn without it, so the cached prefix never recurs and every
    tool round re-writes the whole history (issue #14, second cause).

    What does recur is known without reading native's text: everything through the last
    assistant message, plus the leading blocks of the newest turn that equal the frame Hermes
    queried. The first block native added or changed ends that span. When that block is a
    tool_result (parallel calls, native's reminder on the last result), a breakpoint on the
    unchanged results before it measured no cache hit even though they replay byte-identical
    (#33, cause unknown), so the span ends at the preceding assistant message.
    Recognized native account reminders are first normalized against the host frame.
    The breakpoint never moves later within that normalized frame; unparseable payloads
    forward as is."""
    if not queried:
        return payload
    try:
        body = json.loads(payload)
        messages = body['messages']
        normalized = _normalize_account(messages, queried)
        reordered = _frame_first(messages, queried)
        blocks = [(i, j, b) for i, m in enumerate(messages) if isinstance(m.get('content'), list)
                  for j, b in enumerate(m['content'])]
        marked = [(i, j, b) for i, j, b in blocks if isinstance(b, dict) and 'cache_control' in b]
        if len(marked) != 1:
            return (json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
                    if normalized else payload)
        last = max((i for i, m in enumerate(messages) if m.get('role') == 'assistant'), default=-1)
        stable = [(i, j, b) for i, j, b in blocks if i <= last]
        newest = messages[last + 1] if last + 1 < len(messages) else {}
        if newest.get('role') == 'user' and isinstance(newest.get('content'), list):
            prefix = []
            for j, (sent, host) in enumerate(zip(newest['content'], queried)):
                if _plain(sent) != _plain(host):
                    break
                prefix.append((last + 1, j, sent))
            rest = newest['content'][len(prefix):]
            if not (rest and isinstance(rest[0], dict) and rest[0].get('type') == 'tool_result'):
                stable += prefix
        target = next(((i, j, b) for i, j, b in reversed(stable)
                       if isinstance(b, dict) and b.get('type') not in UNCACHEABLE), None)
        i, j, block = marked[0]
        if target is not None and (target[0], target[1]) < (i, j):
            target[2]['cache_control'] = block.pop('cache_control')
        elif not (reordered or normalized):
            return payload
        return json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        return payload


class Capture:
    def __init__(self):
        self.message = None
        self.complete = False
        self.pending = ''
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.arguments = {}

    def feed(self, chunk):
        self.pending += self.decoder.decode(chunk)
        while match := re.search(r'\r?\n\r?\n', self.pending):
            frame, self.pending = self.pending[:match.start()], self.pending[match.end():]
            data = '\n'.join(line[5:].lstrip(' ') for line in frame.splitlines() if line.startswith('data:'))
            if data:
                self.event(json.loads(data))

    def event(self, event):
        handler = {
            'message_start': self._start,
            'content_block_start': self._block_start,
            'content_block_delta': self._block_delta,
            'content_block_stop': self._block_stop,
            'message_delta': self._delta,
            'message_stop': self._stop,
        }.get(event['type'])
        if handler:
            handler(event)

    def _start(self, event):
        self.message = copy.deepcopy(event['message'])

    def _block_start(self, event):
        self.message['content'].append(copy.deepcopy(event['content_block']))

    def _block_delta(self, event):
        delta = event['delta']
        block = self.message['content'][event['index']]
        field = {'text_delta':'text', 'thinking_delta':'thinking', 'signature_delta':'signature'}.get(delta['type'])
        if field:
            block[field] = block.get(field, '') + delta[field]
        elif delta['type'] == 'input_json_delta':
            index = event['index']
            self.arguments[index] = self.arguments.get(index, '') + delta['partial_json']
        elif delta['type'] == 'citations_delta':
            block.setdefault('citations', []).append(copy.deepcopy(delta['citation']))

    def _block_stop(self, event):
        index = event['index']
        if index in self.arguments:
            # A no-argument tool call streams one input_json_delta with an empty partial_json.
            raw = self.arguments.pop(index)
            self.message['content'][index]['input'] = json.loads(raw) if raw.strip() else {}

    def _delta(self, event):
        self.message.update(event.get('delta', {}))
        self.message['usage'].update(event.get('usage', {}))

    def _stop(self, event):
        self.complete = bool(self.message and self.message.get('stop_reason') and not self.arguments)


class Admission:
    def __init__(self, upstream, timeout, queried=None):
        self.upstream = urlsplit(upstream)
        self.queried = queried
        host = self.upstream.hostname
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == 'localhost'
        if (self.upstream.scheme != 'https' and not (self.upstream.scheme == 'http' and local)) or not host or self.upstream.username or self.upstream.password or self.upstream.query or self.upstream.fragment:
            raise ValueError('Native upstream must be HTTPS or a loopback HTTP fixture')
        self.timeout = timeout
        self.lock = threading.Lock()
        self.sockets = set()
        self.cancelled = False
        self.used = False
        self.denied = 0
        self.request_id = None
        self.status = None
        self.failure = None
        self.capture = Capture()
        self.error_body = b''
        self.prefix = '/admit/' + secrets.token_urlsafe(32)
        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.server.admission = self
        self.url = f'http://127.0.0.1:{self.server.server_port}' + self.prefix
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval':.05}, daemon=True)
        self.thread.start()

    def error_text(self):
        """The upstream's own message for a non-200 answer, '' when none was captured."""
        text = self.error_body.decode('utf-8', errors='replace')
        try:
            message = json.loads(text)['error']['message']
        except (ValueError, KeyError, TypeError):
            return text
        return message if isinstance(message, str) else text

    def abort(self):
        with self.lock:
            self.cancelled = True
            for sock in self.sockets:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # Peer may have closed between the read and cancellation.

    def close(self):
        self.abort()
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Native authorization and the per-call route must never enter logs.

    def do_POST(self):
        gate = self.server.admission
        path = urlsplit(self.path)
        if path.path != gate.prefix + '/v1/messages' or self.headers.get('Origin'):
            self.send_error(404)
            return
        with gate.lock:
            if gate.cancelled or gate.used:
                gate.denied += 1
                body = b'{"type":"error","error":{"type":"invalid_request_error","message":"HERMES_MODEL_ADMISSION_CONSUMED"}}'
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            gate.used = True
            gate.sockets.add(self.connection)
        conn = None
        upstream_socket = None
        try:
            self.connection.settimeout(gate.timeout)
            payload = self.rfile.read(int(self.headers['Content-Length']))
            payload = pin_message_breakpoint(payload, gate.queried)
            target = gate.upstream
            if target.scheme == 'https':
                conn = http.client.HTTPSConnection(target.hostname, target.port, timeout=gate.timeout, context=ssl.create_default_context())
            else:
                conn = http.client.HTTPConnection(target.hostname, target.port, timeout=gate.timeout)
            conn.connect()
            upstream_socket = conn.sock
            with gate.lock:
                if gate.cancelled:
                    return
                gate.sockets.add(upstream_socket)
            # Native identity is preserved; the body has the bounded normalization above.
            headers = {k:v for k,v in self.headers.items() if k.lower() not in ('host','connection','content-length','transfer-encoding','proxy-authorization','proxy-connection','accept-encoding')}
            headers['Accept-Encoding'] = 'identity'
            route = target.path.rstrip('/') + '/v1/messages' + ('?' + path.query if path.query else '')
            conn.request('POST', route, payload, headers)
            del headers, payload
            response = conn.getresponse()
            gate.request_id = response.getheader('request-id') or response.getheader('x-request-id')
            gate.status = response.status
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in ('connection','transfer-encoding','server','date'):
                    self.send_header(key, value)
            self.send_header('Connection', 'close')
            self.end_headers()
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                if response.status == 200:
                    gate.capture.feed(chunk)
                elif len(gate.error_body) < 65536:
                    gate.error_body += chunk  # The rejection reason ("prompt is too long", "adaptive thinking is not supported"), bounded.
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError, TypeError) as exc:
            gate.failure = type(exc).__name__
        finally:
            with gate.lock:
                gate.sockets.discard(self.connection)
                gate.sockets.discard(upstream_socket)
            if conn:
                conn.close()
            self.close_connection = True

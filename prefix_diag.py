"""Opt-in, content-free prefix diagnostics for cache breaks (#17).

One JSON line per completed request with a short hash and length for every block at three
layers: the OpenAI-shaped messages Hermes passed in (``host``), the stream-json frames this
plugin replays to native (``frames``), and the Messages body the relay forwarded upstream
(``wire``). Comparing consecutive requests of one conversation shows the first block that
stopped replaying byte-identically and which layer changed it first. Nothing else is
recorded: no text, arguments, emails, headers or credentials. Hashes are truncated SHA-256
of canonical JSON; ``cache_control`` is excluded from block hashes (the breakpoint moves by
design) and reported as positions instead.

Enable with ``CLAUDE_SUBSCRIPTION_DIRECTSDK_PREFIX_DIAG=/abs/path.jsonl``, or by creating
``$HERMES_HOME/logs/directsdk-prefix-diag.jsonl`` (delete it to stop). Writing stops when the
file exceeds the size cap. Every failure is swallowed: diagnostics never change a request.
"""
import hashlib
import json
import os
from pathlib import Path
import threading
import time

ENV = 'CLAUDE_SUBSCRIPTION_DIRECTSDK_PREFIX_DIAG'
DEFAULT_NAME = 'directsdk-prefix-diag.jsonl'
MAX_BYTES = 256 * 1024 * 1024
_lock = threading.Lock()


def target():
    """The output path when diagnostics are enabled, else None."""
    explicit = os.environ.get(ENV, '').strip()
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_absolute() else None
    home = os.environ.get('HERMES_HOME') or str(Path.home() / '.hermes')
    path = Path(home) / 'logs' / DEFAULT_NAME
    return path if path.is_file() else None


def _canon(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str)


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k != 'cache_control'}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def fingerprint(value):
    """[hash12, length, hash12 of the first 256 characters] of the canonical form."""
    text = _canon(_strip(value))
    digest = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()[:12]
    return [digest(text), len(text), digest(text[:256])]


def _blocks(content):
    if isinstance(content, list):
        return content
    return [] if content in (None, '') else [content]


def host_layer(messages):
    """Per message: role, fingerprint of everything except reasoning payloads, carrier fingerprint."""
    rows = []
    for message in messages or []:
        if not isinstance(message, dict):
            rows.append(['?', fingerprint(message), None])
            continue
        visible = {k: v for k, v in message.items() if k not in ('reasoning_details', 'reasoning_content', 'reasoning')}
        details = message.get('reasoning_details')
        rows.append([message.get('role'), fingerprint(visible), fingerprint(details) if details else None])
    return rows


def frames_layer(system, frames):
    rows = [['system', fingerprint(system)]]
    for frame in frames or []:
        message = frame.get('message', {}) if isinstance(frame, dict) else {}
        rows.append([frame.get('type') if isinstance(frame, dict) else '?',
                     [[b.get('type') if isinstance(b, dict) else '?', *fingerprint(b)]
                      for b in _blocks(message.get('content'))]])
    return rows


def wire_layer(payload):
    """System, tools and every message block of the forwarded body; None when unparseable."""
    try:
        body = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(body, dict):
        return None
    marks = []
    system = []
    for j, block in enumerate(_blocks(body.get('system'))):
        system.append(fingerprint(block))
        if isinstance(block, dict) and 'cache_control' in block:
            marks.append(['s', j, _canon(block['cache_control'])])
    tools = [fingerprint(tool) for tool in body.get('tools') or []]
    for j, tool in enumerate(body.get('tools') or []):
        if isinstance(tool, dict) and 'cache_control' in tool:
            marks.append(['t', j, _canon(tool['cache_control'])])
    messages = []
    for i, message in enumerate(body.get('messages') or []):
        if not isinstance(message, dict):
            messages.append(['?', []])
            continue
        blocks = []
        for j, block in enumerate(_blocks(message.get('content'))):
            blocks.append([block.get('type') if isinstance(block, dict) else 'str', *fingerprint(block)])
            if isinstance(block, dict) and 'cache_control' in block:
                marks.append(['m', i, j, _canon(block['cache_control'])])
        messages.append([message.get('role'), blocks])
    return {'system': system, 'tools': tools, 'tools_all': fingerprint(body.get('tools') or [])[0],
            'messages': messages, 'marks': marks,
            'other': fingerprint({k: v for k, v in body.items() if k not in ('system', 'tools', 'messages')})}


def write(record):
    try:
        path = target()
        if path is None:
            return
        line = _canon(record) + '\n'
        with _lock:
            try:
                if path.stat().st_size > MAX_BYTES:
                    return
            except FileNotFoundError:
                pass
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line.encode('utf-8'))
            finally:
                os.close(fd)
    except Exception:
        pass  # Diagnostics must never alter a request or its cleanup.


def now():
    return round(time.time(), 3)

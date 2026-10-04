"""Reject wrong-provider ids before native, with Hermes' real retry verdict."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import threading
from unittest.mock import patch

import pytest

from test_directsdk_admission import NATIVE


@pytest.mark.parametrize('model', [
    'gpt-6-astra', 'gpt-5.6-sol', 'gpt-6.1-sol', 'gpt-6-sol',
    'gpt-6-astra[1m]', 'gemini-3-pro', ' GPT-6-ASTRA[1M] ',
])
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_non_claude_fails_once_before_native(profile, tmp_path, model, streaming, asynchronous):
    from agent.error_classifier import FailoverReason, classify_api_error

    upstream = []
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            upstream.append(self.path)
            self.rfile.read(int(self.headers['Content-Length']))
            body = json.dumps({'error': {'message': 'model: ' + model}}).encode()
            self.send_response(404)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    thread.start()
    capture = tmp_path / 'invocations.jsonl'
    native = tmp_path / 'native.py'
    native.write_text(NATIVE.replace(
        'for line in sys.stdin:',
        "import pathlib\nwith pathlib.Path(os.environ['ARGV_CAPTURE']).open('a') as capture:\n"
        "    capture.write(json.dumps(sys.argv) + '\\n')\nfor line in sys.stdin:"))
    client = profile.create_client(command=[sys.executable, str(native)], env={
        'PATH': os.defpath, 'HOME': str(tmp_path), 'ARGV_CAPTURE': str(capture),
        'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{peer.server_port}',
    })
    sdk = sys.modules[type(client).__module__]
    request = dict(model=model, messages=[{'role': 'user', 'content': 'fixture'}], stream=streaming)

    async def call_async():
        result = await client.chat.completions.create(**request)
        if streaming:
            async for _ in result:
                pass

    errors = []
    try:
        with patch.object(sdk, 'Admission', wraps=sdk.Admission) as admission, \
                patch.object(sdk.tempfile, 'TemporaryDirectory', wraps=sdk.tempfile.TemporaryDirectory) as temporary:
            # Drive the real classifier with a three-attempt budget, including a large
            # conversation: a bare status-less RuntimeError would retry every time.
            for _ in range(3):
                try:
                    if asynchronous:
                        asyncio.run(call_async())
                    else:
                        result = client.chat.completions.create(**request)
                        if streaming:
                            list(result)
                except RuntimeError as error:
                    errors.append(error)
                    verdict = classify_api_error(error, provider=profile.name, model=model,
                                                 approx_tokens=150_000, num_messages=100,
                                                 context_length=200_000)
                    if not verdict.retryable:
                        break
                else:
                    pytest.fail('non-Claude request succeeded')
            observed = {
                'errors': len(errors),
                'native': len(capture.read_text().splitlines()) if capture.exists() else 0,
                'admission': admission.call_count,
                'upstream': len(upstream),
                'temporary': temporary.call_count,
            }
            assert observed == {'errors': 1, 'native': 0, 'admission': 0, 'upstream': 0, 'temporary': 0}
            assert type(errors[0]) is sdk.ClaudeAPIError
            assert errors[0].status_code == verdict.status_code == 404
            assert model in str(errors[0]) and profile.name in str(errors[0])
            assert verdict.reason == FailoverReason.model_not_found
            assert not verdict.retryable and not verdict.should_compress
            assert verdict.should_fallback and not verdict.should_rotate_credential
            assert not client._requests and client._owned_cwd is None
    finally:
        client.close()
        peer.shutdown()
        thread.join(timeout=5)
        peer.server_close()
        assert not thread.is_alive()

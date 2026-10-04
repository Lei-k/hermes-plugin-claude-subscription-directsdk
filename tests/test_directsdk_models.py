"""Catalog capacity must agree with the actual native --model selection."""
import json
import os
import sys

import pytest

from test_directsdk import FAKE

EXPECTED = {
    'claude-sonnet-5[1m]': 1_000_000,
    'claude-haiku-4-5-20251001': 200_000,
    'claude-opus-5-5[1m]': 1_000_000,
    'claude-opus-5[1m]': 1_000_000,
    'claude-opus-4-8[1m]': 1_000_000,
    'claude-fable-5-1[1m]': 1_000_000,
}


def test_catalog_windows_match_explicit_native_routes(profile):
    from agent.model_metadata import get_model_context_length
    assert set(profile.fallback_models) == set(EXPECTED)
    assert profile.default_aux_model == 'claude-sonnet-5[1m]'
    for model, window in EXPECTED.items():
        assert profile.get_model_context_length(model) == window
        assert get_model_context_length(model, provider=profile.name) == window
        assert get_model_context_length(model, provider=profile.name, config_context_length=200000) == 200000
    # Unpinned: the plain id runs natively within the 200K gateway default; [1m] promises nothing.
    assert profile.get_model_context_length('unqualified-future-model') == 200_000
    assert profile.get_model_context_length('unqualified-future-model[1m]') is None
    assert profile.get_model_context_length('claude-opus-9-9') == 200_000
    assert get_model_context_length('claude-opus-9-9', provider=profile.name) == 200_000


@pytest.mark.parametrize('streaming', [False, True])
def test_native_argv_enables_only_known_long_context_models(profile, tmp_path, streaming):
    capture = tmp_path / 'argv.json'
    native = tmp_path / 'native.py'
    native.write_text(FAKE.replace('rows=[]', "pathlib.Path(os.environ['ARGV_CAPTURE']).write_text(json.dumps(sys.argv))\nrows=[]"))
    aliases = {'sonnet':'claude-sonnet-5[1m]', 'opus':'claude-opus-5-5[1m]',
               'haiku':'claude-haiku-4-5-20251001', 'fable':'claude-fable-5-1[1m]',
               'sonnet[1m]':'claude-sonnet-5[1m]', 'opus[1m]':'claude-opus-5-5[1m]',
               'fable[1m]':'claude-fable-5-1[1m]', 'claude-haiku-4-5':'claude-haiku-4-5-20251001',
               'claude-opus-9-9':'claude-opus-9-9', 'claude-opus-9-9[1m]':'claude-opus-9-9[1m]'}
    # The guard normalizes a copy; argv retains native_model's existing spelling.
    aliases.update({model: model for model in (
        'default', 'best', 'opusplan', 'Opus', 'SONNET', 'Claude-Opus-5-5', 'opus[1M]',
        ' DEFAULT ', ' Best[1M] ', 'OPUSPLAN[1M]', ' Haiku ', ' Fable[1M] ',
        ' claude-opus-9-9 ', 'Claude-Opus-9-9[1M]',
    )})
    with_client = profile.create_client(command=[sys.executable,str(native)], env={'PATH':os.defpath,'HOME':str(tmp_path),'ARGV_CAPTURE':str(capture)})
    try:
        for requested, expected in {**{m:m for m in EXPECTED}, **aliases}.items():
            result = with_client.create(model=requested, messages=[{'role':'user','content':'fixture'}], stream=streaming,
                               tools=[{'type':'function','function':{'name':'probe','description':'TAIL','parameters':{'type':'object','properties':{'value':{'type':'string'}}}}}])
            if streaming:
                list(result)
            argv = json.loads(capture.read_text())
            assert argv[argv.index('--model')+1] == expected and '--effort' not in argv
    finally:
        with_client.close()


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('model', ['anthropic/claude-opus-5-5', 'anthropic.claude-opus-5-5',
                                 'us.anthropic.claude-opus-5-5', ' Anthropic/Claude-Opus-5-5 '])
def test_vendor_prefixed_claude_id_suggests_dropping_prefix(profile, model, streaming):
    client = profile.create_client(command='unused-offline-native', env={})
    try:
        with pytest.raises(RuntimeError, match='drop the vendor prefix') as raised:
            client.create(model=model, messages=[{'role': 'user', 'content': 'fixture'}], stream=streaming)
        assert raised.value.status_code == 404
        assert model in str(raised.value) and profile.name in str(raised.value)
        assert not client._requests and client._owned_cwd is None
    finally:
        client.close()

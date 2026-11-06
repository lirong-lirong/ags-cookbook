import importlib.util
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('demo', ROOT / 'scripts/demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_policy_file_is_a_default_deny_policy():
    policy = demo.load_policy(ROOT / 'network-policy.json')
    assert policy['DefaultDecision'] == 'deny'
    allow_host, allow_port, deny_host, deny_port = demo.policy_probe_targets(policy)
    assert allow_host == 'github.com' and allow_port == '443'
    assert deny_host == 'api.openai.com' and deny_port == '443'


def test_tool_config_declares_envd_port_and_probe():
    cfg = demo.tool_config('ccr.ccs.tencentyun.com/ags-image/osworld1-base:envd-np-v1')
    assert {'Name': 'envd', 'Port': 49983, 'Protocol': 'TCP'} in cfg['Ports']
    assert cfg['Probe']['HttpGet'] == {'Path': '/health', 'Port': 49983, 'Scheme': 'HTTP'}
    assert cfg['Command'] == ['/sbin/init']
    assert 'Args' not in cfg
    assert 'Env' not in cfg


def test_create_payload_has_no_storage_mounts(tmp_path, monkeypatch):
    d = object.__new__(demo.Trial)
    d.state = {}
    d.path = tmp_path / 'state.json'
    monkeypatch.delenv('ROLE_ARN', raising=False)
    calls = []

    def api(action, **params):
        calls.append((action, params))
        if action == 'CreateSandboxTool':
            return {'ToolId': 'owned-tool'}
        return {'SandboxToolSet': [{'Status': 'ACTIVE'}]}

    d.api = api
    d.create('ccr.ccs.tencentyun.com/ags-image/osworld1-base:envd-np-v1')
    params = calls[0][1]
    assert calls[0][0] == 'CreateSandboxTool'
    assert 'StorageMounts' not in params
    assert params['CustomConfiguration']['Command'] == ['/sbin/init']
    assert {'Name': 'envd', 'Port': 49983, 'Protocol': 'TCP'} in params['CustomConfiguration']['Ports']


def test_start_sends_policy_and_envd_version(tmp_path):
    d = object.__new__(demo.Trial)
    d.state = {'tool_id': 'tool', 'instance_id': 'expired', 'start_token': 'old'}
    d.path = tmp_path / 'state.json'
    calls = []

    def api(action, **params):
        if action == 'DescribeSandboxInstanceList':
            return {'InstanceSet': []}
        calls.append((action, params))
        return {'Instance': {'InstanceId': 'new'}}

    d.api = api
    policy = demo.load_policy(ROOT / 'network-policy.json')
    d.start(policy)
    action, params = calls[-1]
    assert action == 'StartSandboxInstance'
    assert params['NetworkPolicy'] == policy
    assert params['Metadata'] == [{'Name': 'x-envd-version', 'Value': '0.5.14'}]


def test_verify_policy_accepts_status_line_before_download_timeout(tmp_path):
    d = object.__new__(demo.Trial)
    d.state = {'instance_id': 'i-test', 'image': 'img-test'}
    d.path = tmp_path / 'state.json'
    d.region = 'ap-guangzhou'
    d.request = lambda *a, **kw: None
    d.wait_ready = lambda *a, **kw: None
    d.execute = lambda cmd, **kw: {'output': (
        'envd_pid=1\nenvd_user=root\nenvd_local_health=ok\nca_extra=absent\n'
        'probe github.com rc=28 http=200 err=curl: (28) Operation timed out\n'
        'probe api.openai.com rc=0 http=403 err=\n')}
    report = d.verify_policy(demo.load_policy(ROOT / 'network-policy.json'))
    assert report['result'] == 'PASS'
    assert report['deny_evidence'] == 'policy-denied'


def test_load_policy_rejects_invalid_decision(tmp_path):
    path = tmp_path / 'policy.json'
    path.write_text('{"DefaultDecision": "sometimes", "Rules": []}')
    with pytest.raises(ValueError, match='DefaultDecision'):
        demo.load_policy(path)


def test_policy_probe_targets_requires_an_allow_rule():
    with pytest.raises(ValueError, match='allow Rule'):
        demo.policy_probe_targets({'DefaultDecision': 'deny', 'Rules': []})


def _stub_trial(monkeypatch):
    calls = []

    class Stub:
        def __init__(self, *a, **kw):
            pass

        def create(self, image, **kw):
            calls.append(('create', image, kw))

        def start(self, policy):
            calls.append(('start',))

        def verify_policy(self, policy):
            calls.append(('verify',))

        def preview(self):
            calls.append(('preview',))

    monkeypatch.setattr(demo, 'Trial', Stub)
    monkeypatch.setattr(demo, 'load_dotenv', lambda *a, **kw: None)
    monkeypatch.setenv('AGS_REGION', 'ap-guangzhou')
    return calls


def test_run_uses_custom_image_with_special_tag(monkeypatch):
    calls = _stub_trial(monkeypatch)
    image = 'ccr.ccs.tencentyun.com/ags-image/osworld1-base:envd-np-v1'
    monkeypatch.setenv('OSWORLD1_BASE_IMAGE',
                       'ccr.ccs.tencentyun.com/ags-image/osworld1-base:base-tag@sha256:' + 'a' * 64)
    monkeypatch.setenv('CUSTOM_IMAGE_OSWORLD1', image)
    monkeypatch.setattr('sys.argv', ['demo.py', 'run', '--flavor', 'osworld1'])
    demo.main()
    assert [c[0] for c in calls] == ['create', 'start', 'verify', 'preview']
    assert calls[0][1] == image
    assert calls[0][2] == {}


def test_build_uses_envd_dockerfile_and_special_tag(monkeypatch):
    calls = []
    monkeypatch.setattr(demo, 'load_dotenv', lambda *a, **kw: None)
    monkeypatch.setattr(demo.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setenv('OSWORLD2_BASE_IMAGE',
                       'ccr.ccs.tencentyun.com/ags-image/osworld2-base:base-tag@sha256:' + 'b' * 64)
    monkeypatch.setenv('CUSTOM_IMAGE_OSWORLD2',
                       'ccr.ccs.tencentyun.com/ags-image/osworld2-base:envd-np-v1')
    monkeypatch.setenv('CONTAINER_TOOL', 'podman')
    monkeypatch.setattr('sys.argv', ['demo.py', 'build', '--flavor', 'osworld2'])
    demo.main()
    assert calls[0][0] == 'podman'
    assert any('Dockerfile.envd' in str(part) for part in calls[0])
    assert 'ccr.ccs.tencentyun.com/ags-image/osworld2-base:envd-np-v1' in calls[0]


def test_build_rejects_overwriting_base_tag(monkeypatch):
    monkeypatch.setattr(demo, 'load_dotenv', lambda *a, **kw: None)
    monkeypatch.setenv('OSWORLD1_BASE_IMAGE',
                       'ccr.ccs.tencentyun.com/ags-image/osworld1-base:base-tag@sha256:' + 'c' * 64)
    monkeypatch.setenv('CUSTOM_IMAGE_OSWORLD1',
                       'ccr.ccs.tencentyun.com/ags-image/osworld1-base:base-tag')
    monkeypatch.setattr('sys.argv', ['demo.py', 'build', '--flavor', 'osworld1'])
    with pytest.raises(SystemExit):
        demo.main()


def test_clean_with_state_dir_cleans_only_that_state(monkeypatch, tmp_path):
    cleaned = []

    class Stub:
        def __init__(self, state_dir, *a, **kw):
            cleaned.append(state_dir)

        def clean(self):
            pass

    monkeypatch.setattr(demo, 'Trial', Stub)
    monkeypatch.setattr(demo, 'load_dotenv', lambda *a, **kw: None)
    monkeypatch.setenv('AGS_REGION', 'ap-guangzhou')
    monkeypatch.setattr('sys.argv', ['demo.py', 'clean', '--state-dir', str(tmp_path)])
    demo.main()
    assert cleaned == [tmp_path]


def test_novnc_url_authenticates_page_and_websocket():
    token = 'a+b/c?d=e&f%20'
    url = demo.novnc_url('5910-example.ap-guangzhou.tencentags.com', token)
    query = parse_qs(urlsplit(url).query)
    assert query['access_token'] == [token]
    assert parse_qs(urlsplit(query['path'][0]).query)['token'] == [token]

"""OSWorld + root envd + NetworkPolicy: build an image, run a sandbox, verify."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv
import requests
from tencentcloud.ags.v20250920 import ags_client
from tencentcloud.common.credential import Credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

ROOT = Path(__file__).resolve().parents[1]
ENVD_PORT = 49983
ENVD_VERSION = '0.5.14'


def tool_config(image: str) -> dict:
    # Registry type describes the service, not repository visibility.
    registry = image.split('/', 1)[0].lower()
    registry_type = ('personal' if registry.endswith('.ccs.tencentyun.com') else
                     'enterprise' if registry.endswith('.tencentcloudcr.com') else 'custom')
    if '@' in image:
        reference, digest = image.rsplit('@', 1)
        prefix, name = reference.rsplit('/', 1)
        if registry_type == 'personal':
            if ':' not in name:
                raise ValueError('Personal CCR needs an immutable tag; supply tag@sha256:digest')
            image = reference
        else:
            image = prefix + '/' + name.split(':', 1)[0] + '@' + digest
    # The platform only reaches envd (and injects its CA bundle) when 49983 is
    # declared on the Tool; /health is the readiness signal. The image starts
    # envd through its systemd unit, so /sbin/init stays the Tool Command.
    ports = [('osworld', 5000), ('novnc', 5910), ('vlc', 8080), ('cdp', 9222),
             ('envd', ENVD_PORT)]
    return {
        'Image': image,
        'ImageRegistryType': registry_type,
        'Command': ['/sbin/init'],
        'Ports': [{'Name': name, 'Port': port, 'Protocol': 'TCP'} for name, port in ports],
        'Resources': {'CPU': '8', 'Memory': '16Gi', 'Storage': '20Gi'},
        'Probe': {'HttpGet': {'Path': '/health', 'Port': ENVD_PORT, 'Scheme': 'HTTP'},
                  'ReadyTimeoutMs': 30000, 'ProbeTimeoutMs': 3000,
                  'ProbePeriodMs': 1000, 'SuccessThreshold': 1, 'FailureThreshold': 100},
    }


def novnc_url(host: str, token: str) -> str:
    # Authenticate both the document request and the subsequent WebSocket.
    ws_path = 'websockify' + ('?' + urlencode({'token': token}) if token else '')
    params = {'autoconnect': 'true', 'resize': 'scale', 'path': ws_path}
    if token:
        params['access_token'] = token
    return f'https://{host}/vnc.html?' + urlencode(params)


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(policy, dict) or not isinstance(policy.get('Rules'), list):
        raise ValueError('NetworkPolicy must be a JSON object with a Rules array')
    if policy.get('DefaultDecision', 'deny') not in {'allow', 'deny'}:
        raise ValueError('DefaultDecision must be allow or deny')
    for rule in policy['Rules']:
        if not isinstance(rule, dict) or rule.get('Decision') not in {'allow', 'deny', 'upstream'}:
            raise ValueError('Every Rule needs a Decision of allow/deny/upstream')
        if not rule.get('Resources'):
            raise ValueError('Every Rule needs at least one Resource')
    return policy


def policy_probe_targets(policy: dict) -> tuple[str, str, str, str]:
    """Pick one concrete allow host and one concrete deny host from the policy."""
    def first(decision: str) -> tuple[str, str] | None:
        for rule in policy['Rules']:
            if rule.get('Decision') != decision:
                continue
            for resource in rule['Resources']:
                hosts = [h for h in resource.get('Hosts', []) if '*' not in h]
                if hosts:
                    ports = resource.get('Ports') or ['443']
                    return hosts[0], str(ports[0])
        return None

    allow = first('allow')
    deny = first('deny')
    if not allow:
        raise ValueError('Policy needs an allow Rule with a concrete Host to probe')
    if not deny and policy.get('DefaultDecision', 'deny') != 'deny':
        raise ValueError('A fully-allow policy has no deny target to probe')
    return (*allow, *(deny or ('example.com', '443')))


class Trial:
    def __init__(self, state_dir: Path | None = None, region: str | None = None):
        self.region = region or os.getenv('AGS_REGION', 'ap-guangzhou')
        self.path = (state_dir or ROOT / '.state' / self.region) / 'state.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}
        http = HttpProfile(endpoint='ags.tencentcloudapi.com', reqTimeout=180)
        profile = ClientProfile(httpProfile=http)
        secret_id = os.getenv('TENCENTCLOUD_SECRET_ID')
        secret_key = os.getenv('TENCENTCLOUD_SECRET_KEY')
        if not secret_id or not secret_key:
            raise ValueError('Set TENCENTCLOUD_SECRET_ID and TENCENTCLOUD_SECRET_KEY')
        # The SDK EnvironmentVariableCredential omits STS session tokens.
        credential = Credential(secret_id, secret_key, os.getenv('TENCENTCLOUD_TOKEN'))
        self.client = ags_client.AgsClient(credential, self.region, profile)

    def api(self, action: str, **params) -> dict:
        # call_json preserves recently added API fields even when typed models lag.
        return self.client.call_json(action, params)['Response']

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2) + '\n')

    def tool(self) -> dict:
        rows = self.api('DescribeSandboxToolList', ToolIds=[self.state['tool_id']], Limit=1).get('SandboxToolSet', [])
        if not rows:
            raise RuntimeError('Tool is missing. Run make clean before starting again.')
        return rows[0]

    def create(self, image: str):
        if self.state.get('image') and self.state['image'] != image:
            raise RuntimeError('Existing state uses another image; run make clean first.')
        if not self.state.get('tool_id'):
            cfg = tool_config(image)
            name = self.state.setdefault('tool_name', 'osworld-np-' + uuid.uuid4().hex[:12])
            self.state.setdefault('create_token', uuid.uuid4().hex)
            self.state['image'] = image
            self.save()
            payload = dict(ToolName=name, ToolType='custom',
                           Description='OSWorld envd + NetworkPolicy example',
                           DefaultTimeout='1h', NetworkConfiguration={'NetworkMode': 'PUBLIC'},
                           ClientToken=self.state['create_token'], CustomConfiguration=cfg)
            if os.getenv('ROLE_ARN'):
                payload['RoleArn'] = os.environ['ROLE_ARN']
            response = self.api('CreateSandboxTool', **payload)
            self.state.update(tool_id=response['ToolId'], tool_name=name, image=image)
            self.state.pop('create_token', None)
            self.save()  # Record ownership before any polling/network operation.
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            tool = self.tool()
            print(f"Tool {self.state['tool_id']}: {tool.get('Status')}", flush=True)
            if tool.get('Status') == 'ACTIVE':
                if '@' in image:
                    expected = image.rsplit('@', 1)[1]
                    actual = (tool.get('CustomConfiguration') or {}).get('ImageDigest')
                    if actual != expected:
                        raise RuntimeError('Tool manifest digest differs from the expected digest')
                return
            if tool.get('Status') in {'FAILED', 'ERROR', 'DELETED', 'INACTIVE'}:
                raise RuntimeError('Tool preparation failed: ' + str(tool.get('StatusReason', '')))
            time.sleep(10)
        raise TimeoutError('Tool preparation timed out; state retained for inspection/cleanup.')

    def start(self, policy: dict):
        if self.state.get('instance_id'):
            rows = self.api('DescribeSandboxInstanceList', InstanceIds=[self.state['instance_id']], Limit=1).get('InstanceSet', [])
            if rows and rows[0].get('Status') == 'RUNNING':
                return
            if rows and rows[0].get('Status') not in {'STOPPED', 'TERMINATED', 'DELETED', 'EXPIRED'}:
                raise RuntimeError('Existing instance is not running; inspect or clean it before starting another.')
            self.state.pop('instance_id', None)
            self.state.pop('start_token', None)
            self.save()
        mode = os.getenv('AUTH_MODE', 'token').upper()
        if mode not in {'TOKEN', 'NONE'}:
            raise ValueError('AUTH_MODE must be token or none')
        started = time.monotonic()
        # Persist the idempotency key so an ambiguous timeout can be retried safely.
        self.state.setdefault('start_token', uuid.uuid4().hex)
        self.save()
        params = dict(ToolId=self.state['tool_id'], Timeout='1h',
                      AuthMode=mode, ClientToken=self.state['start_token'],
                      NetworkPolicy=policy,
                      Metadata=[{'Name': 'x-envd-version', 'Value': ENVD_VERSION}])
        try:
            r = self.api('StartSandboxInstance', **params)
        except TencentCloudSDKException as exc:
            self.state['last_start_error'] = {'code': exc.get_code(), 'request_id': exc.get_request_id()}
            self.save()
            raise
        self.state.pop('last_start_error', None)
        self.state.update(instance_id=r['Instance']['InstanceId'], auth_mode=mode,
                          start_seconds=round(time.monotonic() - started, 3))
        self.save()
        print(f"Instance {self.state['instance_id']} created in {self.state['start_seconds']}s", flush=True)

    def token(self) -> str:
        if self.state.get('auth_mode') == 'NONE':
            return ''
        return self.api('AcquireSandboxInstanceToken', InstanceId=self.state['instance_id'])['Token']

    def host(self, port: int) -> str:
        return f"{port}-{self.state['instance_id']}.{self.region}.tencentags.com"

    def request(self, path: str, *, port: int = 5000, method='GET', **kwargs):
        token = self.token()
        headers = {'X-Access-Token': token} if token else {}
        r = requests.request(method, f'https://{self.host(port)}{path}', headers=headers,
                             timeout=kwargs.pop('timeout', 60), **kwargs)
        # Do not include request bodies or credential-bearing URLs in errors.
        if not r.ok:
            raise RuntimeError(f'OSWorld {method} :{port}{path}: HTTP {r.status_code}')
        return r

    def execute(self, command: list[str], *, timeout: int = 60) -> dict:
        result = self.request('/execute', method='POST', json={'command': command, 'shell': False},
                              timeout=timeout).json()
        if result.get('returncode') != 0:
            raise RuntimeError('OSWorld command failed (output omitted; may contain credentials)')
        return result

    def wait_ready(self, path: str, port: int = 5000, *, timeout: int = 360) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.request(path, port=port, timeout=10)
                return
            except (RuntimeError, requests.RequestException):
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'{path}:{port} did not become ready within {timeout}s') from None
                time.sleep(3)

    def verify_policy(self, policy: dict) -> dict:
        allow_host, allow_port, deny_host, deny_port = policy_probe_targets(policy)
        # With an envd probe the instance becomes RUNNING before the desktop is up.
        self.wait_ready('/platform')
        try:
            self.request('/health', port=ENVD_PORT, timeout=10)
            data_plane = 'ok'
        except (RuntimeError, requests.RequestException):
            data_plane = 'external-check-unavailable'
        script = f'''
set -u
printf 'envd_pid=%s\\n' "$(pgrep -x envd | head -n 1)"
printf 'envd_user=%s\\n' "$(ps -o user= -C envd | head -n 1 | tr -d ' ')"
printf 'envd_local_health=%s\\n' "$(curl -fsS -m 5 -o /dev/null http://127.0.0.1:{ENVD_PORT}/health && echo ok || echo fail)"
printf 'ca_extra=%s\\n' "$(test -f /usr/local/share/ca-certificates/e2b-ca.crt && echo present || echo absent)"
probe() {{
  host=$1
  port=$2
  err=$(mktemp)
  code=$(curl -sS -m 30 -o /dev/null -w '%{{http_code}}' "https://$host:$port/" 2>"$err")
  rc=$?
  printf 'probe %s rc=%s http=%s err=%s\\n' "$host" "$rc" "$code" "$(head -c 200 "$err" | tr '\\n' ' ')"
  rm -f "$err"
}}
probe {shlex.quote(allow_host)} {shlex.quote(allow_port)}
probe {shlex.quote(deny_host)} {shlex.quote(deny_port)}
exit 0
'''
        output = self.execute(['sh', '-c', script], timeout=120).get('output', '')
        values, probes = {}, []
        for line in output.splitlines():
            if line.startswith('probe '):
                match = re.match(r'probe (\S+) rc=(\d+) http=(\S+) err=(.*)', line)
                if match:
                    probes.append({'host': match.group(1), 'rc': int(match.group(2)),
                                   'http': match.group(3), 'err': match.group(4).strip()})
            elif '=' in line:
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip()
        allow_probe = next((p for p in probes if p['host'] == allow_host), {})
        deny_probe = next((p for p in probes if p['host'] == deny_host), {})
        # A large page can time out after the status line arrived. Any real
        # non-403 status still proves the request passed the egress policy.
        allow_code = str(allow_probe.get('http', ''))
        allow_ok = allow_code.isdigit() and allow_code not in ('000', '403')
        denied = deny_probe.get('http') == '403' or deny_probe.get('rc', 0) != 0
        evidence = deny_probe.get('http', '') + ' ' + deny_probe.get('err', '')
        report = {
            'instance_id': self.state.get('instance_id'),
            'image': self.state.get('image'),
            'region': self.region,
            'envd': {'port': ENVD_PORT, 'data_plane_health': data_plane,
                     'pid': values.get('envd_pid'), 'user': values.get('envd_user'),
                     'local_health': values.get('envd_local_health'),
                     'ca_extra': values.get('ca_extra')},
            'probe_targets': {'allow': f'{allow_host}:{allow_port}', 'deny': f'{deny_host}:{deny_port}'},
            'network_policy': policy,
            'allow_probe': allow_probe,
            'deny_probe': deny_probe,
            'deny_evidence': ('policy-denied' if '403' in evidence else
                              'connection-blocked' if denied else 'not-denied'),
            'result': 'PASS' if (allow_ok and denied and values.get('envd_user') == 'root'
                                 and values.get('envd_local_health') == 'ok') else 'FAIL',
        }
        self.path.with_name('policy-report.json').write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        if report['result'] != 'PASS':
            raise RuntimeError('envd/NetworkPolicy check failed; see policy-report.json')
        return report

    def update_policy(self, policy: dict) -> dict:
        instance_id = self.state.get('instance_id')
        if not instance_id:
            raise RuntimeError('No running instance in state; run make run first')
        self.api('UpdateSandboxInstance', InstanceId=instance_id, NetworkPolicy=policy)
        rows = self.api('DescribeSandboxInstanceList', InstanceIds=[instance_id], Limit=1).get('InstanceSet', [])
        # Update only returns a RequestId; describe the instance to read back.
        readback = (rows[0].get('NetworkPolicy') if rows else None) or {}
        print(json.dumps({'instance_id': instance_id, 'network_policy_readback': readback},
                         indent=2, ensure_ascii=False), flush=True)
        return readback

    def preview(self):
        print('noVNC URL (contains an instance credential; do not share publicly):\n' + novnc_url(self.host(5910), self.token()))
        print('The interactive instance remains available for up to one hour. Run make clean when finished.')

    def stop(self):
        if self.state.get('instance_id'):
            try:
                self.api('StopSandboxInstance', InstanceId=self.state['instance_id'])
            except TencentCloudSDKException as exc:
                if not (exc.get_code() or '').startswith('ResourceNotFound'):
                    raise
                rows = self.api('DescribeSandboxInstanceList', InstanceIds=[self.state['instance_id']], Limit=1).get('InstanceSet', [])
                if rows:
                    raise
            self.state.pop('instance_id', None)
            self.state.pop('start_token', None)
            self.save()

    def clean(self):
        self.stop()
        if self.state.get('create_token') and not self.state.get('tool_id'):
            raise RuntimeError('Tool creation result is unknown; retry the same command to recover its ID before cleanup.')
        if self.state.get('start_token') and not self.state.get('instance_id'):
            raise RuntimeError('Start result is unknown; state retained. Retry the same command or inspect instances before deleting the Tool.')
        if self.state.get('tool_id'):
            self.api('DeleteSandboxTool', ToolId=self.state['tool_id'])
            self.state = {}
            self.save()


def main():
    load_dotenv(ROOT / '.env', override=False)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['build', 'push', 'run', 'verify', 'update', 'clean'])
    p.add_argument('--flavor', choices=['osworld1', 'osworld2'], default='osworld1',
                   help='Which of the two OSWorld base images to use')
    p.add_argument('--policy-file', type=Path, help='NetworkPolicy JSON (default: network-policy.json)')
    p.add_argument('--state-dir', type=Path)
    args = p.parse_args()
    suffix = 'OSWORLD1' if args.flavor == 'osworld1' else 'OSWORLD2'
    base = os.getenv(f'{suffix}_BASE_IMAGE', '')
    custom = os.getenv(f'CUSTOM_IMAGE_{suffix}', '')
    container_tool = os.getenv('CONTAINER_TOOL', 'docker')
    if args.action in {'build', 'push'}:
        if not base or not custom:
            p.error(f'Set {suffix}_BASE_IMAGE and CUSTOM_IMAGE_{suffix} in .env')
        # The envd image is pushed to the base image's own repository under a
        # dedicated tag; reusing the base tag would overwrite the upstream image.
        if custom == base.split('@', 1)[0]:
            p.error('CUSTOM_IMAGE_* must use a dedicated tag; never overwrite the base image tag')
        cmd = [container_tool, 'push', custom] if args.action == 'push' else [
            container_tool, 'build', '--platform=linux/amd64',
            '--build-arg', 'OSWORLD_BASE_IMAGE=' + base,
            '-f', str(ROOT / 'Dockerfile.envd'), '-t', custom, str(ROOT)]
        subprocess.run(cmd, check=True)
        return
    policy = None
    if args.action in {'run', 'verify', 'update'}:
        policy_file = args.policy_file or Path(os.getenv('NETWORK_POLICY_FILE') or ROOT / 'network-policy.json')
        policy = load_policy(policy_file)
    region = os.getenv('AGS_REGION', 'ap-guangzhou')
    state_dir = args.state_dir or ROOT / '.state' / region / args.flavor
    if args.action == 'clean':
        if args.state_dir:
            Trial(args.state_dir).clean()
        else:
            for name in ('osworld1', 'osworld2'):
                Trial(ROOT / '.state' / region / name).clean()
        return
    trial = Trial(state_dir)
    if args.action == 'verify':
        trial.verify_policy(policy)
    elif args.action == 'update':
        trial.update_policy(policy)
        trial.verify_policy(policy)
    else:
        if not custom:
            p.error(f'Set CUSTOM_IMAGE_{suffix} in .env')
        trial.create(custom)
        trial.start(policy)
        trial.verify_policy(policy)
        trial.preview()


if __name__ == '__main__':
    main()

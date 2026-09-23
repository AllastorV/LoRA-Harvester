"""Stdlib local client and portable configuration snippets. No API keys required."""
from __future__ import annotations
import http.client
import json
import time
import uuid
from pathlib import Path
from .ai_control import ControlError, TERMINAL


class LiveClient:
    def __init__(self, session_file):
        self.session_file = Path(session_file)

    def request(self, method, path, body=None):
        try:
            if self.session_file.is_symlink() or self.session_file.stat().st_size > 4096:
                raise ControlError('session', 'Invalid local session file.')
            data = json.loads(self.session_file.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ControlError('not_connected', 'Open LoRA Harvester, click the AI status button, add approved folders, and enable the connection.') from exc
        if (not isinstance(data, dict) or data.get('version') != 1 or type(data.get('port')) is not int or not 1 <= data['port'] <= 65535
                or not isinstance(data.get('token'), str) or not 32 <= len(data['token']) <= 128):
            raise ControlError('session', 'Invalid session metadata.')
        conn = http.client.HTTPConnection('127.0.0.1', data['port'], timeout=5)
        try:
            payload = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            conn.request(method, path, body=payload, headers={'Authorization': 'Bearer ' + data['token'],
                         'Content-Type': 'application/json', 'Host': f'127.0.0.1:{data["port"]}'})
            response = conn.getresponse()
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ControlError('response_size', 'Response exceeds client budget.')
            result = json.loads(raw)
            if response.status != 200:
                error = result.get('error') or {}
                raise ControlError(error.get('code', 'http'), error.get('message', str(response.status)))
            return result
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise ControlError('connection', 'Local bridge unavailable. Do not retry a write with a new ID; query the same request ID after reconnecting.') from exc
        finally:
            conn.close()

    def call(self, action, arguments, request_id=None, wait=1.5):
        identifier = request_id or uuid.uuid4().hex
        body = {'action': action, 'arguments': arguments, 'request_id': identifier}
        try:
            result = self.request('POST', '/command', body)
        except ControlError as exc:
            error = ControlError(exc.code, str(exc) + ' Request ID: ' + identifier)
            error.request_id = identifier
            raise error from exc
        deadline = time.monotonic() + wait
        while result['status'] not in TERMINAL and result['status'] != 'pending_approval' and time.monotonic() < deadline:
            time.sleep(.05)
            result = self.request('GET', '/requests/' + result['id'])
        return result


def connection_config(root, python):
    root = Path(root).absolute()
    python = str(Path(python).absolute())
    if Path(python).name.lower() == 'pythonw.exe':
        python = str(Path(python).with_name('python.exe'))
    script = str(root / 'mcp_server.py')
    # JSON string literals are valid TOML basic strings for normal Windows paths.
    toml = ('[mcp_servers.lora_harvester]\ncommand = ' + json.dumps(python) + '\nargs = [' +
            json.dumps(script) + ']\nstartup_timeout_sec = 20\ntool_timeout_sec = 30\n')
    claude = {'mcpServers': {'lora_harvester': {'command': python, 'args': [script]}}}
    return ('# Codex: ilgili config.toml dosyasına bu bloğu ekle\n' + toml +
            '\n# Claude Code / Desktop: ilgili MCP JSON ayarına yalnız bu sunucuyu birleştir\n' +
            json.dumps(claude, ensure_ascii=False, indent=2) + '\n')

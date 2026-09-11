"""Run the installed public web command and verify the files it actually serves."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


def run_web(project, out, env, timeout=1800):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    cli = project / 'node_modules/@bloomengine/engine/tools/cli/bloom.cjs'
    command = [shutil.which('node'), str(cli), 'run', '--web', '--port', str(port)]
    record = dict(command=command, cwd=str(project), status='running', served_files={})
    def save(): (out / 'installed-web-run.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    save()
    started = time.monotonic()
    with (out / 'installed-web-run.stdout.log').open('wb') as stdout, (out / 'installed-web-run.stderr.log').open('wb') as stderr:
        process = subprocess.Popen(command, cwd=project, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=os.name != 'nt',
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            while time.monotonic() - started < timeout:
                if process.poll() is not None:
                    raise RuntimeError(f'installed bloom run --web exited before serving: {process.returncode}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError('installed bloom run --web did not serve before timeout')
            for name in ('index.html', 'assets/welcome.txt', 'assets_manifest.json', 'pkg/bloom_web_bg.wasm'):
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/{name}', timeout=15) as response:
                    data = response.read()
                    mime = response.headers.get_content_type()
                if data != (project / 'dist/web' / name).read_bytes():
                    raise RuntimeError(f'installed server returned different bytes for {name}')
                if name.endswith('.wasm') and mime != 'application/wasm':
                    raise RuntimeError('installed server did not serve the engine with the WASM MIME type')
                record['served_files'][name] = dict(sha256=hashlib.sha256(data).hexdigest(), mime=mime)
            record['status'] = 'pass'
            record['duration_seconds'] = round(time.monotonic() - started, 3)
            save()
            return record
        except Exception as error:
            record.update(status='fail', error=str(error), duration_seconds=round(time.monotonic() - started, 3))
            save()
            raise
        finally:
            # Terminate only this invocation's owned process tree; a failure may
            # happen while its compiler child is still running.
            if process.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name != 'nt': os.killpg(process.pid, signal.SIGKILL)
                    else: process.kill()
                    process.wait(timeout=5)

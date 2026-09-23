"""Isolated AI Toolkit execution; never installs/imports ML dependencies in Harvester.

The selected toolkit checkout and interpreter are trusted executable code. This
is process isolation for dependency/lifecycle management, not a security sandbox.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

from .ai_toolkit_config import ToolkitOptions, _path, _cancelled, json_bytes, validate_bundle

EVENT_PREFIX = 'LH_AITK_EVENT '
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def interpreter_candidates(root: Path) -> list[Path]:
    """Only project-local candidates; never fall back to PATH/Harvester Python."""
    return [root / env / script for env in ('venv', '.venv')
            for script in ('Scripts/python.exe', 'bin/python')
            if (root / env / script).is_file()]


def validate_checkout(root: str, python: str) -> tuple[Path, Path]:
    folder = _path(root, 'AI Toolkit', exists=True)
    # A Unix venv interpreter is normally a symlink to the system binary: allow
    # that *one* file while verifying sys.prefix in the selected process below.
    executable = Path(python).expanduser().absolute()
    if not executable.is_file() or any(c in str(executable) for c in '\x00\r\n'):
        raise ValueError('AI Toolkit Python bulunamadı / interpreter not found.')
    for relative in ('run.py', 'toolkit/config_modules.py', 'toolkit/job.py',
                     'toolkit/data_loader.py', 'jobs/process/BaseSDTrainProcess.py'):
        f = _path(str(folder / relative), 'Toolkit source', exists=True)
        if not f.is_file():
            raise ValueError(f'AI Toolkit eksik / missing: {relative}')
    run_tree = ast.parse((folder / 'run.py').read_text('utf-8-sig'))
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'main'
               for n in run_tree.body):
        raise ValueError('Bu Toolkit CLI sürümü desteklenmiyor / missing run.main.')
    source = (folder / 'toolkit/config_modules.py').read_text('utf-8-sig')
    for key in ('gradient_accumulation', 'dataset_path', 'keep_tokens'):
        if key not in source:
            raise ValueError(f'Toolkit güncellemesi gerekli / missing field: {key}')
    train_src = (folder / 'jobs/process/BaseSDTrainProcess.py').read_text('utf-8-sig')
    if 'def end_step_hook(' not in train_src or 'self.end_step_hook()' not in train_src:
        raise ValueError('Toolkit adım sonu kontrolünü desteklemiyor / no step-boundary hook.')
    return folder, executable


def training_environment(options: ToolkitOptions) -> dict[str, str]:
    env = os.environ.copy()
    # Avoid accidentally importing Harvester modules / startup customizations.
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(name, None)
    env.update(ACCELERATE_TORCH_DEVICE=f'cuda:{options.device_index}', ACCELERATE_USE_CPU='false',
               ACCELERATE_USE_DEEPSPEED='false', ACCELERATE_USE_FSDP='false',
               PYTHONUNBUFFERED='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               SEED=str(options.seed), HF_HUB_DISABLE_TELEMETRY='1',
               DO_NOT_TRACK='1', WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false')
    if not options.allow_downloads:
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    # Opt-in does not override an administrator's existing offline variables.
    return env


PROBE = r'''
import importlib.util, json, sys
mods = ['torch','yaml','diffusers','transformers','accelerate','safetensors']
missing = [n for n in mods if importlib.util.find_spec(n) is None]
r = {'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
     'python': sys.version.split()[0], 'missing': missing, 'cuda': False, 'devices': []}
if 'torch' not in missing:
    import torch
    r['torch'] = torch.__version__
    r['cuda'] = torch.cuda.is_available()
    if r['cuda']:
        r['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        r['bf16'] = [bool(torch.cuda.is_bf16_supported())]  # selected device checked in launcher
r['bitsandbytes'] = importlib.util.find_spec('bitsandbytes') is not None
print('LH_AITK_PROBE ' + json.dumps(r, ensure_ascii=True), flush=True)
'''


def check_runtime(root: str, python: str, options: ToolkitOptions | None = None,
                  cancel: threading.Event | None = None) -> dict:
    folder, executable = validate_checkout(root, python)
    # The probe never loads a model or downloads packages. It may initialize CUDA.
    env = training_environment(options or ToolkitOptions('.', '.', 'unused/repo'))
    proc = subprocess.Popen([str(executable), '-u', '-c', PROBE], cwd=str(folder), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace',
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    deadline = time.monotonic() + 90
    output = ''
    try:
        while True:
            _cancelled(cancel)
            try:
                output, _ = proc.communicate(timeout=.2)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Toolkit ortam kontrolü zaman aşımı / runtime probe timeout.')
    finally:
        if proc.poll() is None:
            proc.kill()  # Probe only: no training / checkpoint write is running.
            proc.communicate()
    records = [line[len('LH_AITK_PROBE '):] for line in output.splitlines()
               if line.startswith('LH_AITK_PROBE ')]
    if proc.returncode != 0 or not records:
        raise RuntimeError('AI Toolkit Python kontrolü başarısız / runtime probe failed:\n' + output[-6000:])
    report = json.loads(records[-1])
    if report['prefix'] == report['base_prefix']:
        raise ValueError('AI Toolkit için ayrı venv Python seç / select a dedicated Toolkit venv.')
    if Path(report['prefix']).resolve() == Path(sys.prefix).resolve():
        raise ValueError('Harvester Python ortamını kullanma / Toolkit needs a separate environment.')
    if report['missing']:
        raise ValueError('Toolkit ortamında eksik paketler / missing: ' + ', '.join(report['missing']))
    if not report['cuda']:
        raise ValueError('Seçili Toolkit Python ortamında CUDA çalışmıyor / CUDA unavailable.')
    if options:
        if options.device_index >= len(report['devices']):
            raise ValueError('CUDA aygıt numarası bulunamadı / invalid device index.')
        if options.optimizer == 'adamw8bit' and not report['bitsandbytes']:
            raise ValueError('AdamW8bit için Toolkit ortamında bitsandbytes gerekli.')
    report['toolkit_root'] = str(folder)
    report['executable'] = str(executable)
    return report


@dataclass
class RunResult:
    status: str
    returncode: int | None
    message: str
    log_path: str = ''
    output_path: str = ''
    step: int = 0


class ToolkitRunner:
    """Blocking run called from a QThread. Stop is a cooperative checkpoint-safe request."""
    def __init__(self):
        self.cancel = threading.Event()
        self.process = None
        self.stop_path: Path | None = None
        self._guard = threading.Lock()
        self._forced = False

    def stop(self):
        self.cancel.set()
        with self._guard:
            if self.stop_path is not None:
                self.stop_path.touch(exist_ok=True)

    def force_stop(self):
        """Only after separate UI confirmation. Can interrupt a checkpoint write."""
        self._forced = True
        self.stop()
        if self.process is not None and self.process.poll() is None:
            self.process.kill()

    def run(self, root: str, python: str, manifest_path: Path, *, resume=False,
            log=lambda text: None, progress=lambda event: None) -> RunResult:
        from src.core.clothing_io import FileLock
        self._forced = False
        _cancelled(self.cancel)
        manifest_path = _path(str(manifest_path), 'Job manifest', exists=True)
        if manifest_path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError('Job manifest too large.')
        manifest = json.loads(manifest_path.read_text('utf-8'))
        if manifest.get('schema_version') != 1 or manifest.get('engine') != 'ai-toolkit':
            raise ValueError('Unsupported job manifest.')
        options = ToolkitOptions(**manifest['options'])
        options.validate()
        report = check_runtime(root, python, options, self.cancel)
        folder, executable = validate_checkout(root, python)
        _cancelled(self.cancel)
        out = _path(manifest['output_path'], 'Training output')
        lock_file = out.parent / '.lh-toolkit-jobs' / f'{options.name}.lock'
        _path(str(lock_file), 'Training lock')
        marker = out / '.lh-toolkit-run.json'
        with FileLock(lock_file):
            _cancelled(self.cancel)
            # Validate source images/captions only once, after the environment
            # probe and as close as possible to actual launch.
            if validate_bundle(manifest_path, cancel=self.cancel) != manifest:
                raise ValueError('Job changed during runtime check; rebuild.')
            if out.exists() and any(out.iterdir()):
                if not resume:
                    raise ValueError('Çıktı dolu. Yeni ad seç veya Devam/yeniden dene seç / output exists.')
                if marker.is_symlink() or not marker.is_file():
                    raise ValueError('Harvester kaydı olmayan çıktıya devam edilmez / untracked output.')
                saved = json.loads(marker.read_text('utf-8'))
                if saved.get('recipe_signature') != manifest['recipe_signature']:
                    raise ValueError('Devam için model/dataset/ayarlar uyuşmuyor / resume recipe mismatch.')
                if options.steps <= saved.get('last_step', 0):
                    raise ValueError('Hedef step son kayıttan büyük olmalı / increase target steps.')
            else:
                out.mkdir(exist_ok=True)
            # Once accepted, keep this marker even on failure to record provenance.
            marker_state = {'schema_version': 1, 'recipe_signature': manifest['recipe_signature'],
                            'manifest': str(manifest_path), 'last_step': 0, 'status': 'starting',
                            'runtime': report}
            _atomic_json(marker, marker_state)
            run_id = uuid.uuid4().hex
            jobdir = Path(manifest_path).resolve().parent
            logfile = jobdir / f'run-{run_id}.log'
            self.stop_path = jobdir / f'stop-{run_id}.request'
            # stop() may have arrived while the marker was being prepared.
            _cancelled(self.cancel)
            launcher = Path(__file__).resolve().parents[2] / 'scripts' / 'run_ai_toolkit.py'
            command = [str(executable), '-u', str(launcher), '--root', str(folder),
                       '--config', str(jobdir / 'training.yaml'), '--stop-file', str(self.stop_path),
                       '--device', str(options.device_index), '--dtype', options.dtype]
            step, saw_done, saw_stopped = 0, False, False
            consistent_progress = True
            pending, last_emit = [], time.monotonic()
            with logfile.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write('LoRA-Harvester → AI Toolkit\n' + json.dumps(command, ensure_ascii=True) + '\n')
                log(f'AI Toolkit | {report["python"]} | {report["devices"][options.device_index]}')
                self.process = subprocess.Popen(command, cwd=str(folder), env=training_environment(options),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                    errors='replace', bufsize=1, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    # Universal newlines turns tqdm carriage returns into lines.
                    for line in self.process.stdout:
                        clean = ANSI.sub('', line).rstrip('\r\n')
                        if len(clean) > 32768:
                            clean = clean[:32768] + '…'
                        stream.write(clean + '\n')
                        if clean.startswith(EVENT_PREFIX):
                            try:
                                event = json.loads(clean[len(EVENT_PREFIX):])
                                kind = event.get('event')
                                if kind == 'step':
                                    if int(event['total']) != options.steps or not 0 <= int(event['step']) <= options.steps:
                                        consistent_progress = False
                                        self.stop()
                                        pending.append('Toolkit step/total does not match the recipe; stopping.')
                                    else:
                                        step = max(step, int(event['step']))
                                        progress(event)
                                saw_done |= kind == 'done'
                                saw_stopped |= kind == 'stopped'
                            except (ValueError, KeyError, TypeError):
                                pending.append(clean)
                        else:
                            pending.append(clean)
                        now = time.monotonic()
                        if pending and (now - last_emit >= .15 or len(pending) >= 50):
                            log('\n'.join(pending)); pending.clear(); last_emit = now
                    returncode = self.process.wait()
                finally:
                    # No silent orphan if the pipe/handler fails. Request a safe
                    # stop and wait. A hung GPU call requires explicit force-stop.
                    if self.process.poll() is None:
                        self.stop()
                        self.process.wait()
                    self.process.stdout.close()
                    if pending:
                        log('\n'.join(pending))
                    stream.flush()
            status = 'completed' if returncode == 0 and saw_done and step >= options.steps and consistent_progress else 'failed'
            if saw_stopped or self._forced:
                status = 'stopped' if not self._forced else 'forced'
            if not consistent_progress:
                status = 'failed'
            checkpoints = [p for p in out.glob('*.safetensors') if p.is_file() and p.stat().st_size > 0]
            if status == 'completed' and not checkpoints:
                status = 'failed'
                log('Toolkit bitti ancak LoRA .safetensors çıktısı bulunamadı / no checkpoint produced.')
            marker_state.update(status=status, last_step=step, returncode=returncode)
            _atomic_json(marker, marker_state)
            with self._guard:
                self.stop_path = None
            message = {'completed': 'Eğitim tamamlandı / Training completed.',
                       'stopped': 'Adım sınırında durdu. Son kaydedilen checkpoint korunur; kaydedilmemiş adımlar kaybolabilir.',
                       'forced': 'Zorla durduruldu. Son checkpoint bütünlüğünü kontrol et.',
                       'failed': 'Eğitim başarısız; günlüğü kontrol et / Training failed; inspect the log.'}[status]
            return RunResult(status, returncode, message, str(logfile), str(out), step)


def _atomic_json(path: Path, value):
    _path(str(path), 'Run record')
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('xb') as stream:
            stream.write(json_bytes(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

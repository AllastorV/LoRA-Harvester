"""Standard-library-only, project-scoped installation and diagnostics.

Imported by the bootstrap UI before PyQt/Pillow/torch exist. Package operations
always address a verified venv interpreter, never PATH pip. No data is deleted.
"""
from __future__ import annotations
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED = ((3, 10), (3, 13))
TORCH_CHANNELS = {
    'cu124': ('https://download.pytorch.org/whl/cu124', '2.6.0', '0.21.0'),
    'cu118': ('https://download.pytorch.org/whl/cu118', '2.6.0', '0.21.0'),
    'cpu': ('https://download.pytorch.org/whl/cpu', '2.6.0', '0.21.0'),
}
MODULES = {'numpy': 'numpy', 'cv2': 'opencv-python', 'PIL': 'Pillow',
           'PyQt5.QtCore': 'PyQt5', 'torch': 'torch', 'torchvision': 'torchvision',
           'ultralytics': 'ultralytics', 'transformers': 'transformers',
           'yaml': 'pyyaml', 'pandas': 'pandas', 'onnxruntime': 'onnxruntime',
           'imagehash': 'imagehash', 'sklearn': 'scikit-learn', 'easyocr': 'easyocr',
           'huggingface_hub': 'huggingface_hub', 'scenedetect': 'scenedetect',
           'aiofiles': 'aiofiles', 'tqdm': 'tqdm'}

class SetupError(RuntimeError):
    pass

class SetupCancelled(SetupError):
    pass


def python_in(env: Path, windows=None) -> Path:
    windows = os.name == 'nt' if windows is None else windows
    return Path(env) / ('Scripts/python.exe' if windows else 'bin/python')


def _safe_env_dir(root: Path) -> Path:
    root = Path(root).resolve()
    env = root / 'venv'
    # The interpreter inside a POSIX venv may be a symlink. The environment
    # directory itself must not redirect package writes outside this project.
    if env.is_symlink() or env.resolve() != env.absolute():
        raise SetupError('The environment points outside this project; setup stopped for safety.')
    return env


def clean_env():
    env = dict(os.environ)
    for key in ('PYTHONHOME', 'PYTHONPATH', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_REQUIRE_VIRTUALENV='true', PIP_CONFIG_FILE=os.devnull)
    return env


def probe_python(exe: str | Path, timeout=20):
    script = "import sys,struct,json;print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'version':list(sys.version_info[:3]),'bits':struct.calcsize('P')*8}))"
    try:
        p = subprocess.run([str(exe), '-I', '-c', script], capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=timeout, env=clean_env())
        if p.returncode:
            raise SetupError(p.stderr.strip() or 'Python could not be started.')
        info = json.loads(p.stdout.strip().splitlines()[-1])
        if not isinstance(info, dict):
            raise ValueError('Invalid interpreter report')
        return info
    except (OSError, ValueError, subprocess.TimeoutExpired, IndexError) as exc:
        raise SetupError(f'Python check failed: {exc}') from exc


def verified_venv(root=ROOT, *, require_supported=True) -> Path:
    env = _safe_env_dir(Path(root))
    exe = python_in(env)
    if not exe.is_file():
        raise SetupError('Project venv not found. Run Core setup in the wizard.')
    info = probe_python(exe)
    version = tuple(info['version'][:2])
    if info['bits'] != 64 or (require_supported and not SUPPORTED[0] <= version < SUPPORTED[1]):
        raise SetupError('This dependency profile requires 64-bit Python 3.10–3.12.')
    if (Path(info['prefix']).resolve() != env.resolve()
            or Path(info['prefix']).resolve() == Path(info['base_prefix']).resolve()):
        raise SetupError('The interpreter does not belong to this project environment.')
    cfg = env / 'pyvenv.cfg'
    if not cfg.is_file() or any(
            len(bits := line.lower().split('=', 1)) == 2 and
            bits[0].strip() == 'include-system-site-packages' and bits[1].strip() == 'true'
            for line in cfg.read_text(encoding='utf-8').splitlines()):
        raise SetupError('An invalid environment or one sharing system packages cannot be used.')
    return exe


@contextlib.contextmanager
def setup_lock(root, name='.setup.lock'):
    """OS-held lock: automatically released after a crash, no stale lock deletion."""
    if name not in ('.setup.lock', '.runtime.lock'):
        raise SetupError('Invalid lock name.')
    path = Path(root) / name
    if path.is_symlink():
        raise SetupError('Unsafe setup lock.')
    f = path.open('a+b')
    locked = False
    try:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise SetupError('The application or another setup is running. Close it before installing.') from exc
        yield
    finally:
        if locked:
            f.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def diagnose(root=ROOT, log=None):
    """Read-only. CUDA availability is distinguished from installed CUDA packages."""
    root = Path(root).resolve()
    report = {'root': str(root), 'bootstrap': str(sys.executable), 'venv_ok': False,
              'modules': {}, 'cuda_available': False, 'issues': [], 'warnings': []}
    emit = log or (lambda s: None)
    try:
        exe = verified_venv(root, require_supported=False)
        report['venv_ok'] = True
        report['python'] = probe_python(exe)
        if not SUPPORTED[0] <= tuple(report['python']['version'][:2]) < SUPPORTED[1]:
            report['warnings'].append('The current Python is outside the 3.10–3.12 setup range; installed packages will not be changed automatically.')
        # ONNX loads before Qt in the child just as it does in main.py.
        names = ['onnxruntime'] + [n for n in MODULES if n != 'onnxruntime']
        script = '''import importlib,json,sys
report={"modules":{}}
for name in json.loads(sys.argv[1]):
 try:
  m=importlib.import_module(name)
  report["modules"][name]={"ok":True,"version":str(getattr(m,"__version__",getattr(m,"PYQT_VERSION_STR","")))}
 except Exception as exc: report["modules"][name]={"ok":False,"error":str(exc)}
try:
 import torch
 report["cuda_available"]=bool(torch.cuda.is_available())
 report["torch_cuda"]=torch.version.cuda
 if report["cuda_available"]:
  report["gpu"]=torch.cuda.get_device_name(0)
  x=torch.ones(1,device="cuda"); report["cuda_allocation"]=float(x.cpu().item())==1.0
except Exception as exc: report["cuda_error"]=str(exc)
try:
 import importlib.metadata as md
 report["onnx_distributions"]={n:md.version(n) for n in ("onnxruntime","onnxruntime-gpu") if any(d.metadata.get("Name","").lower()==n for d in md.distributions())}
 import onnxruntime as ort
 report["onnx_providers"]=ort.get_available_providers()
except Exception as exc: report["onnx_error"]=str(exc)
print("LH_DIAG:"+json.dumps(report,ensure_ascii=False))'''
        emit('Checking Python, core modules, and CUDA…')
        p = subprocess.run([str(exe), '-I', '-c', script, json.dumps(names)],
                           capture_output=True, text=True, encoding='utf-8', errors='replace',
                           timeout=120, env=clean_env(), cwd=root)
        line = next((s[8:] for s in reversed(p.stdout.splitlines()) if s.startswith('LH_DIAG:')), None)
        if p.returncode or line is None:
            raise SetupError('Module check did not finish: ' + (p.stderr[-1200:] or p.stdout[-1200:]))
        report.update(json.loads(line))
        report['issues'].extend(f'{n}: {d.get("error", "import error")}'
                                for n, d in report['modules'].items() if not d['ok'])
        if len(report.get('onnx_distributions', {})) > 1:
            report['issues'].append('onnxruntime and onnxruntime-gpu are both installed; choose one runtime.')
        check = subprocess.run([str(exe), '-m', 'pip', 'check'], capture_output=True, text=True,
                               encoding='utf-8', errors='replace', timeout=45, env=clean_env())
        report['pip_check'] = check.stdout.strip() or check.stderr.strip()
        if check.returncode:
            for line in report['pip_check'].splitlines():
                line = line.strip()
                if (report.get('onnx_distributions', {}).get('onnxruntime-gpu')
                        and line.lower().startswith('insightface ')
                        and line.lower().endswith('requires onnxruntime, which is not installed.')):
                    report['warnings'].append('InsightFace metadata requires the CPU ONNX distribution name; the GPU runtime provides the module.')
                elif line:
                    report['issues'].append('pip check: ' + line)
    except (SetupError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
        report['issues'].append(str(exc))
    smi = shutil.which('nvidia-smi')
    if smi:
        try:
            p = subprocess.run([smi, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=10)
            report['nvidia_smi'] = p.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            report['warnings'].append('NVIDIA driver check did not finish.')
    report['ffmpeg'] = shutil.which('ffmpeg') or ''
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:11434/api/tags', timeout=3) as r:
            data = json.loads(r.read(2_000_000))
        report['ollama_models'] = [m.get('name', '') for m in data.get('models', [])]
    except (OSError, ValueError):
        report['ollama_models'] = []
        report['warnings'].append('Local Ollama is not running or reachable; it is required for outfit tagging.')
    return report


class SetupManager:
    def __init__(self, root=ROOT, log=None, cancel=None):
        self.root = Path(root).resolve()
        self.log = log or print
        self.cancel = cancel or threading.Event()
        self.log_path = None

    def emit(self, text):
        self.log(text)
        if self.log_path:
            with self.log_path.open('a', encoding='utf-8') as f:
                f.write(text + '\n')

    def check_cancel(self):
        if self.cancel.is_set():
            raise SetupCancelled('Stopped. Completed steps were preserved; the current pip operation was not interrupted.')

    def run(self, args, *, timeout=None):
        self.check_cancel()
        self.emit('> ' + subprocess.list2cmdline([str(a) for a in args]))
        with subprocess.Popen([str(a) for a in args], cwd=self.root, env=clean_env(),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              encoding='utf-8', errors='replace', bufsize=1) as p:
            # Do not kill pip while it is committing packages; cancellation is
            # honored between steps. The UI remains responsive and says so.
            for line in p.stdout:
                self.emit(line.rstrip())
            code = p.wait(timeout=timeout)
        if code:
            raise SetupError(f'Step failed (exit {code}). Check the first error and the log file.')
        self.check_cancel()

    def ensure_env(self, rebuild=False):
        env = _safe_env_dir(self.root)
        if env.exists():
            try:
                return verified_venv(self.root)
            except SetupError:
                if not rebuild:
                    raise SetupError('The existing environment is invalid or incompatible. Select Rebuild to back it up first; no folder is deleted automatically.')
                try:
                    Path(sys.executable).absolute().relative_to(env.absolute())
                except ValueError:
                    pass
                else:
                    raise SetupError('The running environment cannot rebuild itself; start install.bat with a base Python installation.')
                stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
                dest = self.root / ('venv-backup-' + stamp)
                env.rename(dest)
                self.emit('Previous environment backed up: ' + str(dest))
        info = probe_python(sys.executable)
        if not SUPPORTED[0] <= tuple(info['version'][:2]) < SUPPORTED[1] or info['bits'] != 64:
            raise SetupError('Install 64-bit Python 3.10, 3.11, or 3.12, then run install.bat again.')
        self.run([sys.executable, '-I', '-m', 'venv', str(env)])
        return verified_venv(self.root)

    def pip(self, *args):
        exe = verified_venv(self.root)  # Recheck on every mutation.
        self.run([exe, '-m', 'pip', '--isolated', *args])

    def snapshot(self, exe, stamp):
        p = subprocess.run([str(exe), '-m', 'pip', 'freeze'], capture_output=True, text=True,
                           encoding='utf-8', errors='replace', env=clean_env(), timeout=45)
        (self.root / 'logs' / ('packages-before-' + stamp + '.txt')).write_text(
            p.stdout + ('\n' + p.stderr if p.returncode else ''), encoding='utf-8')

    def install(self, components, *, torch_channel='cu124', rebuild=False):
        allowed = {'core', 'gpu', 'onnx_cpu', 'onnx_gpu', 'clothing', 'upscale', 'anime', 'faces'}
        components = set(components)
        if not components or components - allowed:
            raise SetupError('Select a component to install or use valid component names.')
        if {'onnx_cpu', 'onnx_gpu'} <= components:
            raise SetupError('Choose either CPU or GPU for ONNX, not both.')
        if torch_channel not in TORCH_CHANNELS:
            raise SetupError('Unknown PyTorch wheel channel.')
        if 'onnx_gpu' in components and 'gpu' in components and torch_channel != 'cu124':
            raise SetupError('This ONNX GPU profile requires CUDA 12/cuDNN 9. Select CUDA 12.4 or ONNX CPU.')
        with setup_lock(self.root), setup_lock(self.root, '.runtime.lock'):
            logs = self.root / 'logs'
            logs.mkdir(exist_ok=True)
            stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
            self.log_path = logs / ('setup-' + stamp + '.log')
            self.emit('Only the project environment will change. Datasets, configuration, and models are preserved.')
            python_components = components - {'clothing'}
            if python_components:
                if not python_in(_safe_env_dir(self.root)).is_file() or rebuild:
                    components.add('core')
                    self.emit('Core components were added for the new or rebuilt environment.')
                exe = self.ensure_env(rebuild)
                self.snapshot(exe, stamp)
                self.run([exe, '-m', 'ensurepip', '--upgrade'])
                self.pip('install', '--upgrade', 'pip')
                # Fresh install chooses a wheel channel before ultralytics can
                # implicitly install torch. A healthy existing GPU is not replaced.
                torch_probe = subprocess.run([str(exe), '-I', '-c',
                    'import torch,torchvision,torchaudio; print(torch.__version__,torchvision.__version__,torchaudio.__version__,torch.cuda.is_available())'],
                    capture_output=True, text=True, env=clean_env(), timeout=90)
                has_torch = torch_probe.returncode == 0
                torch_v, vision_v = TORCH_CHANNELS[torch_channel][1:]
                wheel_ready = has_torch and torch_probe.stdout.strip().split() == [
                    f'{torch_v}+{torch_channel}', f'{vision_v}+{torch_channel}',
                    f'{torch_v}+{torch_channel}',
                    str(torch_channel != 'cpu')]
                if not has_torch or (('gpu' in components or torch_channel == 'cpu') and not wheel_ready):
                    channel = torch_channel if 'gpu' in components else 'cpu'
                    url, torch_v, vision_v = TORCH_CHANNELS[channel]
                    # CPU and CUDA wheels may share the same public version.
                    # An explicit GPU repair must not accept an installed CPU
                    # wheel merely because torch==2.6.0 is already satisfied.
                    flags = ['--force-reinstall'] if has_torch else []
                    self.pip('install', *flags, '-c', str(self.root / 'requirements-compat.txt'),
                             f'torch=={torch_v}', f'torchvision=={vision_v}',
                             f'torchaudio=={torch_v}', '--index-url', url)
                if 'core' in components:
                    # Keep an already installed runtime. Avoid installing both
                    # distributions, which share the onnxruntime import package.
                    runtime = subprocess.run([str(exe), '-I', '-c',
                        'import importlib.metadata as m; print(m.version("onnxruntime-gpu"))'],
                        capture_output=True, env=clean_env(), timeout=90).returncode == 0
                    if not runtime and not components.intersection({'onnx_cpu', 'onnx_gpu'}):
                        self.pip('install', 'onnxruntime>=1.19,<2')
                    self.pip('install', '-r', str(self.root / 'requirements-core.txt'),
                             '-c', str(self.root / 'requirements-compat.txt'))
                extras = {'upscale': ['realesrgan>=0.3.0', 'basicsr>=1.4.2', 'gfpgan>=1.3.8'],
                          'anime': ['dghs-imgutils'], 'faces': ['insightface>=0.7.3']}
                for component, packages in extras.items():
                    if component in components:
                        self.pip('install', '-c', str(self.root / 'requirements-compat.txt'), *packages)
                # Optional packages can pull CPU ONNX; apply the selected runtime last.
                if components.intersection({'onnx_cpu', 'onnx_gpu'}):
                    # Shared DLL namespace requires removal of both; downloads
                    # are staged first so an offline failure leaves runtime intact.
                    package = 'onnxruntime-gpu==1.20.2' if 'onnx_gpu' in components else 'onnxruntime>=1.19,<2'
                    if 'onnx_gpu' in components:
                        cuda = subprocess.run([str(exe), '-I', '-c',
                            'import torch; print(torch.version.cuda or "cpu")'], capture_output=True, text=True,
                            encoding='utf-8', errors='replace', env=clean_env(), timeout=30)
                        if cuda.returncode or not cuda.stdout.strip().startswith('12.'):
                            raise SetupError('ONNX GPU requires CUDA 12 PyTorch; select ONNX CPU or CUDA 12.4.')
                    wheel_dir = logs / ('onnx-wheels-' + stamp)
                    wheel_dir.mkdir()
                    self.pip('download', '--only-binary=:all:', '--dest', str(wheel_dir), '-c', str(self.root / 'requirements-compat.txt'), package)
                    self.pip('uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu')
                    self.pip('install', '--no-index', '--find-links', str(wheel_dir), '-c', str(self.root / 'requirements-compat.txt'), package)
                    shutil.rmtree(wheel_dir, ignore_errors=True)
                # Final diagnose runs pip check and handles the GPU runtime package alias.
            if 'clothing' in components:
                self.install_clothing()
            self.emit('Running final checks…')
            report = diagnose(self.root, self.emit)
            (logs / ('diagnostics-' + stamp + '.json')).write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            if python_components and report['issues']:
                raise SetupError('Installation steps finished, but final checks found issues:\n' + '\n'.join(report['issues']))
            if 'gpu' in components and torch_channel != 'cpu' and not report.get('cuda_available'):
                raise SetupError('CUDA packages installed, but the GPU is unavailable. Check the NVIDIA driver and report.')
            if 'onnx_gpu' in components and 'CUDAExecutionProvider' not in report.get('onnx_providers', []):
                raise SetupError('The ONNX CUDA provider is unavailable. GPU setup was not marked successful.')
            self.emit('Selected steps completed. Start the program with run.bat.')
            return report

    def install_clothing(self):
        candidates = [shutil.which('ollama'), str(Path(os.environ.get('LOCALAPPDATA', '')) /
                                                 'Programs/Ollama/ollama.exe')]
        exe = next((p for p in candidates if p and Path(p).is_file()), None)
        if not exe:
            raise SetupError('Ollama was not found. Install it from the wizard link, then retry the Outfit component.')
        # A service is not silently started or installed as an administrator.
        check = subprocess.run([exe, 'list'], capture_output=True, text=True, timeout=15)
        if check.returncode:
            raise SetupError('Start Ollama from the Start menu, then retry.')
        self.run([exe, 'pull', 'qwen3-vl:4b'])

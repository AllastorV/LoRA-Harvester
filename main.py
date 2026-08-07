"""
LoRA-Harvester - Main Application
AI-Powered Dataset Collection Tool for LoRA Training
"""

import sys
import os
import traceback
import datetime

# CRITICAL (Windows): pre-load onnxruntime BEFORE PyQt5. PyQt5 ships Qt DLLs
# that, when loaded first, break onnxruntime's native DLL (onnxruntime_pybind11_state),
# which would silently disable WD14 tagging and imgutils anime detection.
# Importing onnxruntime here first fixes the DLL load order for the whole process.
try:
    import onnxruntime as _ort  # noqa: F401
except Exception:
    pass

# Force UTF-8 output so emoji/Turkish chars don't crash on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Crash log — writes every unhandled Python exception to crash_log.txt
_CRASH_LOG = os.path.join(os.path.dirname(__file__), 'crash_log.txt')

def _crash_handler(exc_type, exc_value, exc_traceback):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(_CRASH_LOG, 'a', encoding='utf-8') as f:
            f.write(f'\n{"="*60}\n{ts}\n')
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = _crash_handler

# Windows: set unique App ID so taskbar shows our icon instead of Python's
if sys.platform == 'win32':
    try:
        import ctypes
        from ctypes import wintypes
        _SetAppID = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        _SetAppID.argtypes = [ctypes.c_wchar_p]
        _SetAppID.restype = ctypes.HRESULT
        _SetAppID("LoRA-Harvester.App.2.0")
    except Exception:
        pass

# Windows: ensure CUDA 12 DLLs are discoverable for onnxruntime-gpu
if sys.platform == 'win32':
    _cuda_bin_dirs = []
    _cuda_env = os.environ.get('CUDA_PATH', '')
    if _cuda_env:
        _cuda_bin_dirs.append(os.path.join(_cuda_env, 'bin'))
    for _v in ('v12.1', 'v12.2', 'v12.3', 'v12.4', 'v12.5', 'v12.6'):
        _cuda_bin_dirs.append(rf'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{_v}\bin')
    for _d in _cuda_bin_dirs:
        if os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
            except (OSError, AttributeError):
                pass
            if _d not in os.environ.get('PATH', ''):
                os.environ['PATH'] = _d + os.pathsep + os.environ.get('PATH', '')

# Pre-import onnxruntime before PyTorch to avoid DLL conflicts
try:
    import onnxruntime
except ImportError:
    pass

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.ui.main_window import create_app
except ModuleNotFoundError as _e:
    # Dependencies not installed (users often run run.bat without install.bat).
    _missing = _e.name or str(_e)
    _msg = (
        f"Missing dependency: '{_missing}'\n\n"
        f"Dependencies are not installed for this Python interpreter:\n"
        f"  {sys.executable}\n\n"
        f"Fix: run install.bat (or install_gpu.bat for NVIDIA GPU),\n"
        f"or install manually:\n"
        f'  "{sys.executable}" -m pip install -r requirements.txt'
    )
    print("=" * 60)
    print(f"[ERROR] {_msg}")
    print("=" * 60)
    _has_console = True
    if sys.platform == 'win32':
        try:
            import ctypes
            _has_console = ctypes.windll.kernel32.GetConsoleWindow() != 0
            if not _has_console:
                # Launched without a console (e.g. run_silent.vbs) — show a message box
                ctypes.windll.user32.MessageBoxW(0, _msg, "LoRA-Harvester", 0x10)
        except Exception:
            pass
    if _has_console and sys.stdin and sys.stdin.isatty():
        _answer = input("Install dependencies now? [y/N]: ").strip().lower()
        if _answer == 'y':
            import subprocess
            _req = os.path.join(os.path.dirname(__file__), 'requirements.txt')
            _rc = subprocess.call([sys.executable, '-m', 'pip', 'install', '-r', _req])
            if _rc == 0:
                print("\n[OK] Dependencies installed. Restart the app (run.bat).")
            else:
                print("\n[ERROR] Install failed - see pip output above.")
    sys.exit(1)


def main():
    """Main application entry point"""
    print("="*60)
    print("🌾 LORA-HARVESTER - AI Powered Dataset Collection")
    print("="*60)
    print("🚀 Starting application...")
    print()
    
    # Check CUDA availability
    try:
        import torch
        print("🔍 Checking GPU availability...")
        print(f"   PyTorch version: {torch.__version__}")
        
        # Check CUDA support
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA available: {cuda_available}")
        
        # Try to get CUDA version
        try:
            import torch.version
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                print(f"   CUDA compiled version: {torch.version.cuda}")
        except:
            pass
        
        if cuda_available:
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            print(f"   Current Device: {torch.cuda.current_device()}")
        else:
            print("⚠️  No GPU detected. Possible reasons:")
            print("   1. NVIDIA GPU driver not installed")
            print("   2. CUDA toolkit not installed")
            print("   3. PyTorch installed without CUDA support")
            print("   4. GPU not compatible with CUDA")
            print()
            print("🔧 To fix:")
            print("   1. Install NVIDIA GPU drivers")
            print("   2. Install CUDA toolkit 11.8+")
            print("   3. Reinstall PyTorch with CUDA:")
            print("      pip uninstall torch torchvision")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print()
            print("   Running on CPU (slower but functional)")
    except ImportError:
        print("⚠️  PyTorch not installed. Please install requirements first.")
        print("   Run: pip install -r requirements.txt")
        return
    
    print()
    print("📱 Features:")
    print("   • AI-powered object detection (YOLOv8)")
    print("   • Smart cropping with head space awareness")
    print("   • Automatic subtitle detection & skipping")
    print("   • Multiple vertical formats (9:16, 3:4, 1:1, 4:5)")
    print("   • GPU acceleration support")
    print("   • Drag & drop interface")
    print()
    print("="*60)
    
    # Create and run application
    app, window = create_app()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

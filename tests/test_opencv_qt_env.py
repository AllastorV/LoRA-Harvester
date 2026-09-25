"""opencv-python's bundled Qt must not stop the PyQt5 application from starting."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAVE_QT = importlib.util.find_spec('PyQt5') is not None


@pytest.mark.skipif(not HAVE_QT, reason='PyQt5 unavailable')
def test_only_opencv_qt_overrides_are_removed():
    from src.ui.main_window import drop_opencv_qt_overrides
    cv2_qt = os.path.join('site-packages', 'cv2', 'qt')
    env = {'QT_QPA_PLATFORM_PLUGIN_PATH': os.path.join(cv2_qt, 'plugins'),
           'QT_QPA_FONTDIR': os.path.join(cv2_qt, 'fonts'), 'OTHER': 'kept'}
    drop_opencv_qt_overrides(env)
    assert env == {'OTHER': 'kept'}
    own = {'QT_QPA_PLATFORM_PLUGIN_PATH': os.path.join('opt', 'qt', 'plugins')}
    drop_opencv_qt_overrides(own)
    assert own == {'QT_QPA_PLATFORM_PLUGIN_PATH': os.path.join('opt', 'qt', 'plugins')}


def _cv2_bundles_qt():
    spec = importlib.util.find_spec('cv2')
    return bool(spec and spec.submodule_search_locations and any(
        (Path(p) / 'qt' / 'plugins').is_dir() for p in spec.submodule_search_locations))


def _run_on_xvfb(code):
    # Other tests may have imported cv2 in this process, which exports its Qt paths.
    env = {k: v for k, v in os.environ.items()
           if k not in ('QT_QPA_PLATFORM', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR')}
    return subprocess.run(['xvfb-run', '-a', sys.executable, '-c', code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


@pytest.mark.skipif(not (HAVE_QT and sys.platform.startswith('linux') and shutil.which('xvfb-run')
                         and _cv2_bundles_qt()),
                    reason='needs Linux, xvfb-run and the Qt-enabled opencv-python wheel')
def test_app_starts_on_x11_after_cv2_was_imported():
    show = ('from PyQt5.QtWidgets import QApplication, QLabel\n'
            'app = QApplication.instance() or QApplication([]); w = QLabel("x"); w.show()\n'
            'app.processEvents(); print("started")\n')
    if 'started' not in _run_on_xvfb(show).stdout:
        pytest.skip('Qt cannot open an X11 window here (system xcb libraries missing)')
    result = _run_on_xvfb('import cv2\n'
                          'from src.ui.main_window import drop_opencv_qt_overrides\n'
                          'drop_opencv_qt_overrides()\n' + show)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'started' in result.stdout

import sys, os, ast, pathlib, traceback
sys.path.insert(0, '.')

PASS = "[OK]"; FAIL = "[ERR]"
results = []

def check(label, fn):
    try:
        fn()
        results.append(f"{PASS}  {label}")
    except Exception as e:
        results.append(f"{FAIL}  {label}: {e}")
        traceback.print_exc()

# 1. Syntax
def syntax_check():
    bad = []
    for f in pathlib.Path('.').rglob('*.py'):
        if any(x in str(f) for x in ['__pycache__','venv','.venv','_check.py']): continue
        try: ast.parse(f.read_text(encoding='utf-8',errors='ignore'))
        except SyntaxError as e: bad.append(f"{f}: {e}")
    assert not bad, '\n'.join(bad)
check("Syntax (38 files)", syntax_check)

# 2. Paths
def path_check():
    from src.ui import theme
    assert 'data' in str(theme._PREFS_PATH)
    from src.ui.resource_settings import _SETTINGS_PATH
    assert 'data' in str(_SETTINGS_PATH)
    from src.core.model_paths import MODELS_DIR, WD14_DIR, YOLO_DIR
    assert MODELS_DIR.name == 'models'
    assert pathlib.Path('config/config.yaml').exists()
check("Path references", path_check)

# 3. PyQt5 UI widgets
def qt_check():
    from PyQt5.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)
    from src.ui.caption_studio_page import _GenerateTab
    tab = _GenerateTab(lang='en')
    must = ['model_combo','variant_combo','mode_combo','trigger_edit','suffix_edit',
            'max_tags_spin','conf_spin','wd14_combo','f2_combo','f2_task_combo',
            'start_btn','stop_btn','progress_bar','recursive_cb','overwrite_cb',
            'keep_char_cb','json_cb','wd14_cb','_model_row_widget','preset_combo']
    missing = [a for a in must if not hasattr(tab, a)]
    assert not missing, f"Missing attrs: {missing}"
check("_GenerateTab attrs", qt_check)

# 4. get_settings
def settings_check():
    from PyQt5.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)
    from src.ui.caption_studio_page import _GenerateTab
    tab = _GenerateTab(lang='en')
    s = tab.get_settings()
    required = ['mode','use_wd14','use_florence2','wd14_model',
                'trigger_word','max_tags','min_confidence','florence2_model']
    missing = [k for k in required if k not in s]
    assert not missing, f"Missing keys: {missing}"
check("get_settings keys", settings_check)

# 5. update_language
def lang_check():
    from PyQt5.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)
    from src.ui.caption_studio_page import _GenerateTab
    tab = _GenerateTab(lang='en')
    tab.update_language('tr')
    tab.update_language('en')
check("update_language EN/TR", lang_check)

# 6. Main window
def main_window_check():
    from PyQt5.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)
    from src.ui.main_window import VideoSmartCropperUI
    win = VideoSmartCropperUI()
    win.close()
check("VideoSmartCropperUI init", main_window_check)

# 7. model_installer
def installer_check():
    from src.core.model_installer import ModelInstallThread, GpuInstallThread
    t = ModelInstallThread(include_florence2=False, wd14_repo='SmilingWolf/wd-swinv2-tagger-v3')
    assert hasattr(t, '_wd14_repo')
check("ModelInstallThread", installer_check)

# ── Report ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("  ERROR CHECK REPORT")
print("="*50)
for r in results:
    print(r)
print("="*50)
fails = [r for r in results if FAIL in r]
print(f"\n{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'} ({len(results)} checks)")

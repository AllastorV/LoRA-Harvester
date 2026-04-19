# LoRA-Harvester — Claude Code Session Checkpoint

Read this file at the start of every session. It replaces the need to re-scan the codebase from scratch.

---

## Project Overview

**LoRA-Harvester** is a PyQt5 desktop app for AI/ML dataset creation. It processes video files to extract high-quality frames, applies face/object detection, and exports labeled datasets for LoRA training.

- **UI style**: SaaS dark-mode dashboard (custom theme engine, no Qt Designer)
- **Language**: Python 3.10+, PyQt5
- **Working directory**: `/home/user/LoRA-Harvester`
- **Active branch**: `claude/fix-face-recognition-Drm8e` (local only — no more GitHub pushes needed)
- **Entry point**: `python main.py`

---

## Architecture — UI Files

| File | Role |
|---|---|
| `src/ui/main_window.py` | Main window, sidebar nav, topbar, all pages, processing logic |
| `src/ui/theme.py` | Theme engine: color tokens, stylesheet functions, font scaling, accent system |
| `src/ui/animations.py` | All animation helpers (fade, slide, expand, pulse, hover lift, nav indicator, press flash) |
| `src/ui/resource_settings.py` | Right-side settings drawer: GPU/CPU/RAM sliders, system monitor, accent swatches |
| `src/ui/translations.py` | i18n strings — Turkish (TR) and English (EN) |
| `src/ui/advanced_settings.py` | Collapsible quality/caption/tag settings panels (used inside main_window) |
| `src/ui/caption_studio_page.py` | Caption editor page with danbooru autocomplete |
| `src/ui/character_sort_page.py` | Character recognition/sorting page |
| `src/ui/tag_frequency_page.py` | Tag frequency analyzer page |

---

## Layout Constants

```
Topbar height:  60px
Sidebar width: 240px
Drawer width:  380px
```

---

## Theme System

### Key files: `src/ui/theme.py`

- `fs(base_px)` — font size helper with `_FONT_BASELINE = 1.30` multiplier (makes 130% the "default readable" look)
- `set_theme(mode, font_scale, accent)` — applies theme; persists to `~/.lora_harvester/theme_prefs.json`
- `get_accent()` — returns current accent hex string
- `ACCENT_PRESETS` — 5 preset (color, name) tuples
- All stylesheet functions return f-strings: `btn_action_start()`, `spinbox()`, `slider()`, `combo()`, `global_stylesheet()`, etc.

### Color tokens (examples)
- `BG_WINDOW`, `BG_SURFACE`, `BG_PANEL`, `BG_HOVER`, `BG_ELEVATED`
- `TEXT_PRIMARY`, `TEXT_SECONDARY`, `TEXT_MUTED`
- `ORANGE`, `ORANGE_LIGHT`, `ORANGE_DARK`, `ORANGE_GLOW`, `ORANGE_DIM`, `ORANGE_SUBTLE`
- `BORDER`, `BORDER_LIGHT`, `BORDER_ACCENT`

### Important: `QWidget { background-color: BG_WINDOW }` in `global_stylesheet()` cascades to all children.
**Fix**: Override with `background: transparent` on `QCheckBox` and `QSlider` — already applied.

---

## Animation System

### File: `src/ui/animations.py`

All helpers return the animation object so callers can chain/cancel:

| Helper | Use |
|---|---|
| `fade_in(widget, duration=220)` | Opacity 0→1 |
| `fade_out(widget, duration=180)` | Opacity 1→0, optionally hides |
| `crossfade(old, new, duration=220)` | Parallel fade transition |
| `slide_in(widget, direction, duration=280)` | Slide + fade from direction |
| `smooth_expand(widget, expand, duration=240)` | maxHeight 0↔content (collapsible panels) |
| `progress_smooth(bar, target, duration=260)` | QProgressBar value interpolation |
| `PulseEffect(widget)` | Infinite opacity loop (drop zone hover glow) |
| `HoverLift(button, lift_px=2)` | Event filter: Y nudge on hover |
| `animate_page_switch(stack, old, new)` | Crossfade + 18px slide for page transitions |
| `NavIndicator(parent, color, height=3)` | Sliding underline for active nav button |
| `press_flash(widget, duration=180)` | Opacity dip 1.0→0.6→1.0 for press feedback |

---

## What Has Been Implemented (this session)

### Bug Fixes
- [x] **CMD flash on Windows**: `_NO_WINDOW_KW = {"creationflags": subprocess.CREATE_NO_WINDOW}` added to nvidia-smi subprocess calls in `resource_settings.py`
- [x] **Font sizes too small**: `_FONT_BASELINE = 1.30` in `theme.fs()` — now readable at 1.0× scale
- [x] **Spinbox arrows invisible**: Removed broken SVG data-URI `image:` declarations; use native Qt arrow rendering with 18px-wide `BG_HOVER` buttons
- [x] **Dark backgrounds on checkbox/settings rows**: `background: transparent` added to `QCheckBox` and `QSlider` in `global_stylesheet()`

### UI Relocations
- [x] **System monitor moved to topbar**: `SystemMonitorBar` (horizontal pill layout: CPU • RAM • GPU • VRAM) replaces the old sidebar `SystemMonitorWidget`
- [x] **Monitor removed from settings drawer**: No duplicate monitor in the drawer
- [x] **GPU badge transparent**: `background: transparent; border: none` on `_gpu_badge`
- [x] **Language selector moved to top of drawer** (above GPU section)

### New Features
- [x] **Accent color theming**: 5 preset swatches + custom QColorDialog picker in Theme section of drawer; persists in `~/.lora_harvester/theme_prefs.json`
- [x] **Per-section colors in settings drawer**: Each section (Language, GPU, Batch, CPU, Memory, Misc, Theme) has its own left-border accent color
- [x] **Video trim spinboxes typable**: `setKeyboardTracking(True)`, `setFocusPolicy(Qt.StrongFocus)`, `lineEdit().setReadOnly(False)`, `setMinimumWidth(90)`

### Animations (latest)
- [x] **NavIndicator**: Sliding orange underline follows active sidebar nav button
- [x] **HoverLift**: 2px Y nudge on hover for Start/Pause/Skip/Stop/Browse/Open buttons
- [x] **PulseEffect**: Drop zone pulses while video is held over it during drag
- [x] **Ensemble smooth expand**: Toggling "Ensemble Mode" checkbox smoothly animates the settings group
- [x] **Thumbnail fade-in**: New preview frames fade in when added to the preview grid
- [x] **Progress bar smooth**: Progress value interpolates smoothly instead of jumping
- [x] **Accent swatch press_flash**: Clicking a color swatch triggers opacity dip feedback

---

## Persistence Files

- `~/.lora_harvester/theme_prefs.json` — `{"mode": "dark", "font_scale": 1.0, "accent": "#D97757"}`
- `~/.lora_harvester/resource_settings.json` — GPU/CPU/memory/misc settings including accent copy

---

## Design Principles

1. **SaaS dark-mode aesthetic** — no heavy card borders, transparent components blend into the window
2. **Topbar is transparent** — monitor pills, GPU badge all have `background: transparent; border: none`
3. **Never leak BG_WINDOW** — any new QWidget/QCheckBox/QSlider must explicitly set `background: transparent`
4. **Animations under 300ms** — keeps the UI feeling snappy, not sluggish
5. **No new dependencies** — use only PyQt5 + stdlib + existing requirements.txt

---

## Known Issues / TODO

- NavIndicator position is set via `QTimer.singleShot(0, ...)` to wait for layout; if the sidebar hasn't rendered yet the initial position may be slightly off (one-time issue at startup, corrects on first click)
- Windows CMD flash fix only applies to the nvidia-smi poll in resource_settings.py; any other `subprocess.run()` calls elsewhere should also get `**_NO_WINDOW_KW` if they appear on Windows

---

## Quick Commands

```bash
# Run the app
python main.py

# Syntax check all UI files
python3 -c "import ast; [ast.parse(open(p).read()) for p in ['src/ui/main_window.py','src/ui/resource_settings.py','src/ui/theme.py','src/ui/animations.py']]"

# Git status
git status
git log --oneline -8
```

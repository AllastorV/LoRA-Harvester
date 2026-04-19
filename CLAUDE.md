# LoRA-Harvester — Claude Code Checkpoint

---

## Quick Summary

**LoRA-Harvester** = PyQt5 video-to-dataset tool. SaaS dark-mode UI with animations, accent theming, transparent components, live system monitor. Working on branch `claude/fix-face-recognition-Drm8e`.

**Key folders:**
- `src/ui/` — all UI code (main_window, theme, animations, resource_settings, etc.)
- `src/core/` — processing logic (video frames, face detection, exports)

**Main files:**
- `main_window.py` (1700 lines) — window, sidebar, topbar, pages, buttons
- `theme.py` (1070 lines) — colors, fonts, stylesheets, accent system
- `animations.py` (390 lines) — fade, slide, expand, pulse, hover, nav indicator, press flash
- `resource_settings.py` (1016 lines) — settings drawer, GPU/CPU sliders, system monitor, accent swatches
- `translations.py` — EN/TR i18n

---

## All Changes Done (from start to now)

### 1️⃣ CMD Flash Fix
- **Problem**: Windows console window flashes on every `subprocess.run()` call
- **Fix**: Added `_NO_WINDOW_KW = {"creationflags": subprocess.CREATE_NO_WINDOW}` to nvidia-smi polling in `resource_settings.py`
- **File**: `resource_settings.py` line ~280

### 2️⃣ System Monitor Relocation
- **Was**: Monitor widget at bottom of sidebar + duplicated in settings drawer
- **Now**: `SystemMonitorBar` (horizontal pills: CPU • RAM • GPU • VRAM) sits in topbar left
- **Files**: `main_window.py` (lines 554–590), `resource_settings.py` (lines 392–523)

### 3️⃣ Font Baseline Increase
- **Problem**: UI text unreadable at 1.0× scale; only readable at 1.3×
- **Fix**: `_FONT_BASELINE = 1.30` in `theme.fs()` — now 1.0× is the comfortable default
- **File**: `theme.py` line ~128

### 4️⃣ GPU Badge Transparency
- **Was**: Had background + border (looked like a card)
- **Now**: `background: transparent; border: none; padding: 0 4px;`
- **File**: `main_window.py` line ~555

### 5️⃣ Spinbox Arrow Fix
- **Problem**: Arrow buttons were invisible (SVG data-URIs don't render in Qt stylesheets)
- **Fix**: Removed `image:` declarations; use 18px-wide buttons with `BG_HOVER` background so Qt renders native arrows
- **File**: `theme.py` line ~543

### 6️⃣ Checkbox/Slider Dark Backgrounds
- **Problem**: Global `QWidget { background-color: BG_WINDOW }` leaked to checkboxes and sliders
- **Fix**: Added `background: transparent` to `QCheckBox` and `QSlider` rules
- **File**: `theme.py` lines 250–254, 589–591

### 7️⃣ Drawer Polish
- **Language selector**: Moved to top of drawer (above GPU section)
- **Section colors**: Each drawer section (Language, GPU, Batch, CPU, Memory, Misc, Theme) has distinct left-border accent color
- **Video trim spinboxes**: Made typable with `setKeyboardTracking(True)`, `setFocusPolicy(Qt.StrongFocus)`, `lineEdit().setReadOnly(False)`
- **Files**: `main_window.py` (lines 854–870), `resource_settings.py` (lines 606–700)

### 8️⃣ Accent Color Theming
- **Added**: 5 preset color swatches (orange, red, pink, purple, cyan) + custom color picker (QColorDialog)
- **Persists**: Accent saved to `~/.lora_harvester/theme_prefs.json`, loaded on startup
- **Per-section colors**: Each drawer section border derives from the selected accent
- **Files**: `resource_settings.py` (lines 668–876), `theme.py` (accent system)

### 9️⃣ UI Animations (Latest)
- **NavIndicator**: Orange underline slides smoothly between sidebar nav buttons on page switch
- **HoverLift**: All action buttons (Start, Pause, Skip, Stop, Browse, Open) nudge up 2px on hover
- **PulseEffect**: Drop zone glows/pulses while a file is held over it during drag
- **Ensemble smooth expand**: Toggling "Ensemble Mode" checkbox smoothly animates settings group open/closed
- **Thumbnail fade-in**: New preview frames fade in softly instead of snapping in
- **Progress bar smooth**: Value interpolates over 160ms instead of jumping
- **Accent swatch press_flash**: Clicking a color swatch triggers opacity dip (1.0 → 0.6 → 1.0)
- **Files**: `animations.py` (added `press_flash()` helper), `main_window.py` (wired all 6 sites), `resource_settings.py` (swatch feedback)

### 🔟 Session Checkpoint (CLAUDE.md)
- **Created**: `CLAUDE.md` at project root
- **Purpose**: Persistent session memory — lists architecture, all changes done, design principles, constants, quick commands
- **Auto-read**: Claude Code reads this at session start (no need to re-scan codebase)

---

## Layout Constants

- **Topbar height**: 60px
- **Sidebar width**: 240px  
- **Drawer width**: 380px

---

## Design Rules

1. SaaS dark aesthetic — no card borders, transparent blending
2. Topbar/sidebar always transparent (`background: transparent; border: none`)
3. Never let global `QWidget { background }` leak — override with `background: transparent`
4. Animations ≤ 300ms (keep snappy feel)
5. No new dependencies (PyQt5 + stdlib only)

---

## Color System (`theme.py`)

- **Backgrounds**: `BG_WINDOW`, `BG_SURFACE`, `BG_PANEL`, `BG_HOVER`, `BG_ELEVATED`
- **Text**: `TEXT_PRIMARY`, `TEXT_SECONDARY`, `TEXT_MUTED`
- **Accent**: `ORANGE`, `ORANGE_LIGHT`, `ORANGE_DARK`, `ORANGE_GLOW`, `ORANGE_DIM`, `ORANGE_SUBTLE`
- **Borders**: `BORDER`, `BORDER_LIGHT`, `BORDER_ACCENT`

---

## Animation Helpers (`animations.py`)

| Name | What |
|---|---|
| `fade_in/fade_out` | Opacity transitions |
| `smooth_expand` | Collapse/expand panels |
| `progress_smooth` | Interpolate progress bar |
| `PulseEffect` | Infinite opacity glow |
| `HoverLift` | Y nudge on hover |
| `animate_page_switch` | Crossfade + slide pages |
| `NavIndicator` | Sliding underline |
| `press_flash` | Opacity dip feedback |

---

## Git / Persistence

- **Working branch**: `claude/fix-face-recognition-Drm8e` (local only)
- **Theme**: `~/.lora_harvester/theme_prefs.json` (`mode`, `font_scale`, `accent`)
- **Settings**: `~/.lora_harvester/resource_settings.json` (GPU/CPU/RAM/misc settings)

---

## Run / Test

```bash
python main.py              # Launch app
python3 -m pytest           # Run tests (if any)
git log --oneline -10       # Recent commits
```

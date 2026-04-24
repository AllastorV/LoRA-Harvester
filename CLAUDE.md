# LoRA-Harvester — Claude Code Checkpoint

---

## Quick Summary

**LoRA-Harvester** = PyQt5 video-to-dataset tool. SaaS dark-mode UI with animations, accent theming, transparent components, live system monitor. Branch `claude/fix-face-recognition-Drm8e`.

**Project root (Desktop):**
`C:\Users\cavas\Desktop\LoRA-Harvester-claude-fix-face-recognition-Drm8e\LoRA-Harvester-claude-fix-face-recognition-Drm8e`

**Key folders:**
- `src/ui/` — all UI code (main_window, theme, animations, resource_settings, caption_studio_page, etc.)
- `src/core/` — processing logic (video frames, face detection, exports, advanced_captioner)

**Main files (approximate line counts after all changes):**
- `main_window.py` (~1950 lines) — window, sidebar, topbar, pages, buttons, status system
- `theme.py` (1070 lines) — colors, fonts, stylesheets, accent system
- `animations.py` (~1050 lines) — all animation classes and helpers
- `resource_settings.py` (1016 lines) — settings drawer, GPU/CPU sliders, system monitor, accent swatches
- `caption_studio_page.py` (~1450 lines) — Caption Studio page: Generate + Edit tabs
- `translations.py` — EN/TR i18n

---

## All Changes Done (cumulative)

### 1️⃣ CMD Flash Fix
- **Fix**: `_NO_WINDOW_KW = {"creationflags": subprocess.CREATE_NO_WINDOW}` for nvidia-smi polling
- **File**: `resource_settings.py` ~line 280

### 2️⃣ System Monitor Relocation
- **Now**: `SystemMonitorBar` (CPU • RAM • GPU • VRAM pills) in topbar left
- **Files**: `main_window.py` lines 554–590, `resource_settings.py` lines 392–523

### 3️⃣ Font Baseline Increase
- **Fix**: `_FONT_BASELINE = 1.30` in `theme.fs()` — 1.0× is comfortable default now
- **File**: `theme.py` ~line 128

### 4️⃣ GPU Badge Transparency
- **Fix**: `background: transparent; border: none; padding: 0 4px;`
- **File**: `main_window.py` ~line 555

### 5️⃣ Spinbox Arrow Fix
- **Fix**: Removed SVG `image:` declarations; 18px-wide buttons use `BG_HOVER` so Qt renders native arrows
- **File**: `theme.py` ~line 543

### 6️⃣ Checkbox/Slider Dark Backgrounds
- **Fix**: `background: transparent` added to `QCheckBox` and `QSlider` rules
- **File**: `theme.py` lines 250–254, 589–591

### 7️⃣ Drawer Polish
- Language selector moved to top of drawer; section left-border accent colors; trim spinboxes typable
- **Files**: `main_window.py` lines 854–870, `resource_settings.py` lines 606–700

### 8️⃣ Accent Color Theming
- 5 preset swatches + custom QColorDialog; persisted to `~/.lora_harvester/theme_prefs.json`
- **Files**: `resource_settings.py` lines 668–876, `theme.py`

### 9️⃣ UI Animations (Session 2)
- `NavIndicator`, `HoverLift`, `PulseEffect`, `smooth_expand`, thumbnail fade-in, progress smooth, `press_flash`
- New classes in `animations.py`: `StatusDot`, `RippleButton`, `ToastNotification`, `LoadingSpinner`,
  `SkeletonShimmer`, `ProgressGlow`, `ShimmerLabel`, `TypewriterLog`, `SidebarPulse`
- New helpers: `count_up`, `shake_widget`, `scale_pop`, `stagger_fade_in`, `glitch_effect`, `badge_bounce`
- **Files**: `animations.py`, `main_window.py`, `resource_settings.py`

### 🔟 Status + Stats System (Session 2)
- `StatusDot` in topbar (heartbeat pulse while processing); `_status_label` in topbar
- Stat cards row (Queued / Extracted / Saved) with `count_up` animations
- `_set_status(state)` single method drives dot + label + SidebarPulse
- `_reset_stat(key)` / `_bump_stat(key, n)` helpers
- `on_progress` bumps extracted/saved; `on_finished` syncs final totals + toast; `on_error` glitch + toast
- `update_ui_texts` now re-translates status label + stat card descs on language change
- **File**: `main_window.py`

### 1️⃣1️⃣ Default Language Fix
- `self.update_ui_texts()` called at end of `init_ui` — fixes EN selected but TR text showing
- **File**: `main_window.py` ~line 690

### 1️⃣2️⃣ Emoji Duplication Fix
- Removed emojis from TR/EN translation keys for `page_video_processing` etc.; emojis live only in `update_ui_texts` _nav_labels tuple
- **File**: `translations.py`

### 1️⃣3️⃣ Tag Mode Dropdown (Session 3)
- Added `tag_first` mode (tags on line 1, Florence-2 caption on line 2) to Caption Studio dropdown
- Dropdown now: Tag Only → Tag First → Natural Language → Combined
- Widen visibility gate: `tag_first` shows both WD14 and Florence-2 settings
- `_run_with_florence2` branches: `tags_only`, `tag_first` (newline-joined), `florence2`, `combined`
- CLI `choices` expanded: `['tags_only', 'tag_first', 'florence2', 'combined']` (both `cli.py` + `scripts/cli.py`)
- **Files**: `caption_studio_page.py`, `translations.py` (EN+TR), `cli.py`, `scripts/cli.py`

### 1️⃣4️⃣ Dynamic Title Animation (Session 3)
- `ShimmerLabel` no longer starts on startup — driven by `_set_status()`
- Starts shimmer on `'processing'`, stops on all other states (idle/paused/done/error)
- No flicker: `stop_shimmer()` calls `self.update()` for clean final frame
- **Files**: `main_window.py` (removed unconditional `start_shimmer()`, added hook in `_set_status`), `animations.py`

### 1️⃣5️⃣ Light Mode Title Compatibility (Session 3)
- `ShimmerLabel.set_colors(base, highlight)` public method added to `animations.py`
- `_refresh_all_styles()` calls `self._brand_label.set_colors(theme.TEXT_PRIMARY, theme.ORANGE_LIGHT)` after each theme change
- Dark: TEXT_PRIMARY `#f2efe8` / Light: `#1a1918` — both readable
- **Files**: `animations.py`, `main_window.py`

### 1️⃣6️⃣ Caption→Edit Folder Auto-Sync (Session 3)
- Single line added in `CaptionStudioPage._init_ui`:
  `self.generate_tab.folder_changed.connect(self.edit_tab.reload_folder)`
- Selecting a folder on Generate tab now immediately populates Edit tab (no need to finish captioning)
- **File**: `caption_studio_page.py` ~line 1430

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
6. Read CLAUDE.md instead of scanning the whole codebase each session

---

## Key API / Signal Hooks

| Where | Signal / Method | What it does |
|---|---|---|
| `main_window.py:730` | `_set_status(state)` | Drives dot + label + SidebarPulse + ShimmerLabel |
| `main_window.py:754` | `_reset_stat(key)` | Resets a stat card to "0" |
| `main_window.py:763` | `_bump_stat(key, n)` | Animates stat card up to n |
| `main_window.py:714` | `_update_video_badge(n)` | Shows/hides queue badge on nav button |
| `caption_studio_page.py:383` | `folder_changed = pyqtSignal(str)` | Fires on browse/drop in Generate tab |
| `caption_studio_page.py:1198` | `edit_tab.reload_folder(folder)` | Public; loads folder in Edit tab |

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
| `StatusDot` | Heartbeat dot; `.set_state(state)`, `._state` attr |
| `RippleButton` | Material ripple on click |
| `ToastNotification` | Slide-in toast (bottom-right); `raise_()` called in init |
| `ProgressGlow` | Pulsing drop shadow on progress bar; `.start()/.stop()` |
| `ShimmerLabel` | Travelling gradient text; `set_colors(base,hi)`, `start/stop_shimmer()` |
| `SidebarPulse` | Sidebar nav button glow pulse |
| `count_up` | Animated number increment |
| `shake_widget` | Horizontal shake for validation error |
| `scale_pop` | Scale bounce feedback |
| `glitch_effect` | Rapid flicker for error state |
| `badge_bounce` | Bounce a pill/badge widget |
| `stagger_fade_in` | Sequential fade across a list of widgets |

---

## Color System (`theme.py`)

- **Backgrounds**: `BG_WINDOW`, `BG_SURFACE`, `BG_PANEL`, `BG_HOVER`, `BG_ELEVATED`
- **Text**: `TEXT_PRIMARY`, `TEXT_SECONDARY`, `TEXT_MUTED`
- **Accent**: `ORANGE`, `ORANGE_LIGHT`, `ORANGE_DARK`, `ORANGE_GLOW`, `ORANGE_DIM`, `ORANGE_SUBTLE`
- **Borders**: `BORDER`, `BORDER_LIGHT`, `BORDER_ACCENT`
- **Status**: `RED` (`#e5534b` dark / `#d4443c` light), `GREEN` (in palette)
- `theme.get_accent()` → current accent hex; `theme.set_theme(mode, scale, accent)` → applies + saves

---

## Git / Persistence

- **Working branch**: `claude/fix-face-recognition-Drm8e` (local only)
- **Theme**: `~/.lora_harvester/theme_prefs.json` (`mode`, `font_scale`, `accent`)
- **Settings**: `~/.lora_harvester/resource_settings.json` (GPU/CPU/RAM/misc)

---

## Run / Test

```bash
python main.py              # Launch app
python -c "import ast; ast.parse(open('src/ui/main_window.py',encoding='utf-8').read()); print('OK')"
git log --oneline -10
```

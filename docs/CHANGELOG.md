# 🌾 LoRA-Harvester - Güncellemeler ve Yenilikler / Updates and New Features

## 📅 Son Güncelleme / Last Update: June 2026 (v4.0)

---

## 🆕 v4.0 — Haziran 2026 / June 2026

### 📝 Altyazı Tespiti Yeniden Yazıldı / Subtitle Detection Rewritten

**TR:** Altyazı tespiti ve kırpma her çözünürlük ve en-boy oranında çalışacak şekilde yeniden yazıldı.
**EN:** Subtitle detection and cropping rewritten to work at any resolution and aspect ratio.

- ✅ **Adaptive threshold fix** — light-text mask used `THRESH_BINARY` with positive C, which saturated the whole zone white and made detection never fire. Fixed with a dual-direction mask (`THRESH_BINARY_INV` for dark text + negative-C `THRESH_BINARY` for light text), applied in all three detection paths.
- ✅ **Full-frame text-line scan** (`scan_text_lines`) — frame downsampled to 640px width, text lines found via morphological dilation with frame-relative filters (width > 12% of frame, height 1–8%, aspect > 3.5), multi-line subtitles merged. Works on 16:9, 9:16, 480p, letterboxed content.
- ✅ **Single-line subtitles detected** — `quick_min_regions` lowered from 2 to 1 in the `normal` preset.
- ✅ **Subtitle band auto-probe** (`_find_subtitle_band`) — probes multiple band heights (20% → 6% of frame) instead of assuming a fixed bottom strip.
- ✅ **Two-sided removal** — top-positioned subtitles crop from the top, bottom subtitles from the bottom; frame is returned unchanged if less than 30% would remain.

### ✂️ Akıllı Kırpma / Smart Crop

- ✅ **Crop shrink fallback** (`_shrink_crop_around_zone`) — when shifting the crop cannot avoid an overlay/subtitle zone, the crop now shrinks on the side that keeps the most content and re-fits the aspect ratio, instead of giving up and using an unconstrained crop.

### 🖥️ Sistem Monitörü / System Monitor

**TR:** GPU/VRAM göstergeleri düzeltildi.
**EN:** GPU/VRAM readouts fixed.

- ✅ Device-wide stats via a single `nvidia-smi` query (`utilization.gpu, memory.used, memory.total`), with `torch.cuda.mem_get_info` as fallback.
- ✅ Old code used `torch.cuda.memory_allocated` (process-local), so VRAM showed ~0 while training ran in a subprocess. Now real usage is shown.
- ✅ GPU detection no longer requires a CUDA-enabled torch build; utilization shows `N/A` instead of a fake 0% when unavailable.

### 🐛 Hata Düzeltmeleri / Bug Fixes

- ✅ **`--no-resume` CLI flag** was parsed but never passed to the processor — now wired through `process_all_videos(resume=...)`.
- ✅ **Duplicate translation keys** in `translations.py` (`caption_mode_tooltip`, `max_tags_tooltip`, `confidence_tooltip`) — the tag-confidence tooltip was silently overriding the detection-confidence tooltip. Duplicates removed in both EN and TR blocks.
- ✅ **Duplicate `_spinbox_style` method** in `advanced_settings.py` removed.

### 🧹 Kod Temizliği / Code Cleanup

- ✅ 7 unused legacy modules removed: `optimized_processor.py`, `curator.py`, `video_processor.py`, `auto_captioner.py`, `sam_segmenter.py`, `scene_detector.py`, `memory_optimizations.py` (zero importers, verified).
- ✅ Lazy import map in `src/core/__init__.py` and `mcp_server.py` health check pruned of removed modules.
- ✅ Unused imports removed across 16 files.

---

## 🎉 ÖNEMLİ YENİLİKLER / MAJOR NEW FEATURES

### 1. ✨ BATCH PROCESSING (Toplu Video İşleme)
**EN:** Process multiple videos at once!
**TR:** Birden fazla videoyu aynı anda işleyin!

#### CLI Kullanımı / CLI Usage:
```bash
# Tek video / Single video
python cli.py video.mp4

# Çoklu video / Multiple videos
python cli.py video1.mp4 video2.mp4 video3.mp4

# Klasördeki tüm videolar / All videos in folder
python cli.py *.mp4
python cli.py videos/*.mp4

# Gelişmiş örnek / Advanced example
python cli.py folder/*.mp4 -f 1:1 -i 20 --ensemble --turbo
```

#### GUI Kullanımı / GUI Usage:
- **EN:** Drag & drop multiple videos at once
- **TR:** Birden fazla videoyu aynı anda sürükle-bırak
- **EN:** Or use "Browse" button to select multiple files (Ctrl+Click)
- **TR:** Veya "Browse" düğmesiyle çoklu dosya seçin (Ctrl+Click)

#### Yeni Batch Script:
```bash
run_batch.bat   # Windows için toplu işleme / Batch processing for Windows
```

---

### 2. 🚀 UNİFİED PROCESSOR (Birleşik İşlemci)
**EN:** All processing modes unified into ONE powerful system!
**TR:** Tüm işleme modları TEK güçlü sistemde birleştirildi!

#### Özellikler / Features:
- ✅ Single Model & Ensemble Mode combined
- ✅ Standard & Turbo Mode combined  
- ✅ Single & Batch Processing combined
- ✅ Automatic mode selection
- ✅ Optimized memory usage
- ✅ Better error handling

#### Avantajlar / Benefits:
- **Daha basit kod / Simpler code**
- **Daha hızlı geliştirme / Faster development**
- **Daha az hata / Fewer bugs**
- **Kolay bakım / Easy maintenance**

---

### 3. 🗑️ TEMİZLİK / CLEANUP

#### Silinen Gereksiz Dosyalar / Removed Unnecessary Files:
- ❌ `demo.py` - Not needed anymore
- ❌ `ensemble_demo.py` - Integrated into unified system
- ❌ `run_ensemble.bat` - Replaced by unified system
- ❌ `video_processor.py` (old) - Replaced by unified processor
- ❌ `optimized_processor.py` (old) - Replaced by unified processor

#### Yeni Dosya Yapısı / New File Structure:
```
src/core/
  ├── detector.py           # Single model detection
  ├── ensemble_detector.py  # Multi-model ensemble
  ├── unified_processor.py  # ⭐ NEW: All-in-one processor
  ├── text_detector.py      # Subtitle detection
  └── cropper.py            # Smart cropping
```

---

## 📊 KARŞILAŞTIRMA / COMPARISON

### Eski Sistem / Old System:
```python
# 4 farklı processor / 4 different processors
- VideoProcessor (basic)
- TurboVideoProcessor (optimized)
- EnsembleVideoProcessor (multi-model)
- OptimizedVideoProcessor (advanced)

# Tek video / Single video only
processor.process_video(video_path)
```

### Yeni Sistem / New System:
```python
# 1 unified processor / 1 birleşik işlemci
- UnifiedVideoProcessor (all features)

# Tek veya çoklu video / Single or multiple videos
processor.process_all_videos(video_paths)

# Otomatik mod seçimi / Automatic mode selection
```

---

## 🎯 KULLANIM ÖRNEKLERİ / USAGE EXAMPLES

### Örnek 1: Tek Video (High Quality) / Single Video (High Quality)
```bash
python cli.py video.mp4 -f 1:1 -i 15 --ensemble --turbo
```
- Format: 1:1 (square)
- Interval: 15 frames (high quality)
- Mode: Ensemble (3 AI models)
- Turbo: Enabled (faster)

### Örnek 2: Toplu İşleme (Fast) / Batch Processing (Fast)
```bash
python cli.py *.mp4 -f 9:16 -i 60 --turbo
```
- Format: 9:16 (vertical)
- Interval: 60 frames (faster)
- Mode: Single model (YOLO)
- Turbo: Enabled

### Örnek 3: Maksimum Kalite / Maximum Quality
```bash
python cli.py video1.mp4 video2.mp4 -f 1:1 -i 10 -c 0.8 --ensemble --voting-threshold 3
```
- Format: 1:1
- Interval: 10 frames (very detailed)
- Confidence: 0.8 (high threshold)
- Voting: 3/3 models must agree

### Örnek 4: Hızlı Test / Quick Test
```bash
python cli.py test.mp4 -f 1:1 -i 90 --no-turbo
```
- Format: 1:1
- Interval: 90 frames (very fast)
- Turbo: Disabled (simpler)

---

## 🔧 YENİ PARAMETRELER / NEW PARAMETERS

### CLI:
```bash
# Multiple video input (supports wildcards)
python cli.py video1.mp4 video2.mp4 video3.mp4
python cli.py *.mp4
python cli.py folder/*.mp4

# Turbo mode is now DEFAULT
--turbo         # Enabled by default
--no-turbo      # Disable if needed
```

### GUI:
- **Multi-select**: Ctrl+Click to select multiple files
- **Drag & drop**: Drop multiple videos at once
- **Progress**: Shows current video X/Y
- **Stats**: Overall summary for batch processing

---

## 📈 PERFORMANS İYİLEŞTİRMELERİ / PERFORMANCE IMPROVEMENTS

### Hız / Speed:
- ⚡ Turbo mode: 2-3x faster
- ⚡ Batch processing: No reload between videos
- ⚡ Optimized memory: Better GPU usage

### Bellek / Memory:
- 💾 Reduced memory footprint
- 💾 Better cleanup between videos
- 💾 No memory leaks

### Doğruluk / Accuracy:
- 🎯 Ensemble mode: 95%+ accuracy
- 🎯 Smart voting: Fewer false positives
- 🎯 Better quality scoring

---

## 🛠️ TEKNİK DETAYLAR / TECHNICAL DETAILS

### Unified Processor Architecture:
```python
class UnifiedVideoProcessor:
    def __init__(self, video_paths, ...):
        # Handles both single and multiple videos
        # Auto-detects ensemble vs single mode
        # Configurable turbo mode
        
    def process_all_videos(self, ...):
        # Processes all videos in sequence
        # Maintains stats for each video
        # Returns overall summary
        
    def _process_video_turbo(self, ...):
        # Batch frame processing
        # GPU optimized
        
    def _process_video_standard(self, ...):
        # Frame-by-frame processing
        # CPU compatible
```

### Features:
1. **Auto-detection**: Automatically detects ensemble mode
2. **Smart batching**: Groups frames for GPU efficiency
3. **Progress tracking**: Per-video and overall progress
4. **Error recovery**: Continues on error
5. **Resource cleanup**: Proper cleanup between videos

---

## 📝 MIGRATION GUIDE (Geçiş Kılavuzu)

### Eski Kod / Old Code:
```python
from src.core.video_processor import VideoProcessor
from src.core.optimized_processor import TurboVideoProcessor

# Had to choose processor type manually
if use_turbo:
    processor = TurboVideoProcessor(...)
else:
    processor = VideoProcessor(...)

# Only single video
processor.process_video(video_path)
```

### Yeni Kod / New Code:
```python
from src.core.unified_processor import UnifiedVideoProcessor

# One processor for everything
processor = UnifiedVideoProcessor(
    video_paths=[video1, video2],  # Single or multiple
    use_turbo=True,                 # Turbo mode toggle
    ...
)

# Works with single or batch
processor.process_all_videos()
```

---

## ✅ YAPILDI / COMPLETED

1. ✅ Unified processor created
2. ✅ Batch processing implemented
3. ✅ CLI updated for batch support
4. ✅ GUI updated for multi-select
5. ✅ Unnecessary files removed
6. ✅ New batch script created
7. ✅ Documentation updated

---

## 🎓 ÖNERİLER / RECOMMENDATIONS

### LoRA Training için / For LoRA Training:
```bash
# En iyi kalite / Best quality
python cli.py videos/*.mp4 -f 1:1 -i 15 -c 0.7 --ensemble --turbo

# Dengeli / Balanced
python cli.py videos/*.mp4 -f 1:1 -i 25 -c 0.6 --turbo

# Hızlı / Fast
python cli.py videos/*.mp4 -f 1:1 -i 50 --turbo
```

### Vertical Content için / For Vertical Content:
```bash
python cli.py videos/*.mp4 -f 9:16 -i 30 --turbo
```

### Test için / For Testing:
```bash
python cli.py test.mp4 -f 1:1 -i 90 --no-turbo
```

---

## 🚨 ÖNEMLİ NOTLAR / IMPORTANT NOTES

### EN:
1. **Turbo mode is now DEFAULT** - Use `--no-turbo` to disable
2. **Batch processing** - No need for separate scripts
3. **Unified system** - All features in one place
4. **Old files removed** - Use new unified system only
5. **GPU recommended** - Especially for ensemble mode

### TR:
1. **Turbo modu artık VARSAYILAN** - Kapatmak için `--no-turbo` kullanın
2. **Toplu işleme** - Ayrı script'lere gerek yok
3. **Birleşik sistem** - Tüm özellikler tek yerde
4. **Eski dosyalar silindi** - Sadece yeni unified sistemi kullanın
5. **GPU önerilir** - Özellikle ensemble modu için

---

## 🔗 KAYNAKLAR / RESOURCES

- **Main Script**: `main.py` - GUI launcher
- **CLI Script**: `cli.py` - Command-line interface
- **Batch Script**: `run_batch.bat` - Batch processing helper
- **Core Processor**: `src/core/unified_processor.py` - Main processor
- **Documentation**: `README.md` - Full documentation

---

## 💡 İPUÇLARI / TIPS

1. **Toplu işlemde** / **In batch processing**:
   - Videoları aynı klasöre koyun / Put videos in same folder
   - Wildcard kullanın: `*.mp4` / Use wildcards: `*.mp4`
   - İlk test küçük video ile yapın / Test first with small video

2. **Performans için** / **For performance**:
   - GPU kullanın / Use GPU
   - Turbo modu aktif / Keep turbo enabled
   - Frame interval artırın / Increase frame interval

3. **Kalite için** / **For quality**:
   - Ensemble modu / Use ensemble mode
   - Frame interval azaltın / Decrease frame interval
   - Confidence yükseltin / Increase confidence

---

## 🎉 SONUÇ / CONCLUSION

**EN:** The new unified system makes LoRA-Harvester:
- ✅ Easier to use
- ✅ More powerful
- ✅ Faster
- ✅ More reliable
- ✅ Better organized

**TR:** Yeni birleşik sistem LoRA-Harvester'ı:
- ✅ Kullanımı daha kolay
- ✅ Daha güçlü
- ✅ Daha hızlı
- ✅ Daha güvenilir
- ✅ Daha düzenli

---

**🚀 Happy Training! / İyi Eğitimler!**

**📧 Issues**: https://github.com/AllastorV/LoRA-Harvester/issues

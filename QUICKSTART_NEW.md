# 🚀 LoRA-Harvester - Quick Start Guide

## ⚡ HIZLI BAŞLANGIÇ / QUICK START

### 1️⃣ GUI Mode (Basit / Simple)
```bash
python main.py
```
- Drag & drop multiple videos
- Click "Browse" to select videos (Ctrl+Click for multiple)
- Adjust settings
- Click "Start Processing"

### 2️⃣ CLI Mode (Gelişmiş / Advanced)

#### Tek Video / Single Video:
```bash
python cli.py video.mp4
```

#### Çoklu Video / Multiple Videos:
```bash
# Method 1: List videos
python cli.py video1.mp4 video2.mp4 video3.mp4

# Method 2: Use wildcards
python cli.py *.mp4
python cli.py videos/*.mp4

# Method 3: Batch script
run_batch.bat
```

---

## 🎯 ÖNERILEN AYARLAR / RECOMMENDED SETTINGS

### 🏆 LoRA Training (Best Quality)
```bash
python cli.py videos/*.mp4 -f 1:1 -i 15 -c 0.7 --ensemble --turbo
```
**Sonuç / Result**: ~500-1000 frames per 10-minute video

### ⚡ Fast Collection
```bash
python cli.py videos/*.mp4 -f 1:1 -i 50 --turbo
```
**Sonuç / Result**: ~200-400 frames per 10-minute video

### 📱 Vertical Content (TikTok/Reels)
```bash
python cli.py videos/*.mp4 -f 9:16 -i 30 --turbo
```
**Sonuç / Result**: Optimized for vertical platforms

---

## 📋 PARAMETREler / PARAMETERS

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `-f` | Format (1:1, 9:16, 3:4, 4:5, 16:9, 4:3) | 9:16 | **1:1** for LoRA |
| `-i` | Frame interval (lower = more frames) | 30 | **15-25** for quality |
| `-c` | Confidence (0.1-0.95) | 0.5 | **0.6-0.7** for quality |
| `-p` | Padding (pixels) | 500 | 500 |
| `--ensemble` | Use 3 AI models | OFF | **ON** for quality |
| `--turbo` | Batch processing | **ON** | Keep ON |
| `--voting-threshold` | Min model agreements (1-3) | 2 | 2 for balanced, 3 for strict |

---

## 💡 SENARYOLAR / SCENARIOS

### Scenario 1: Karakter LoRA Eğitimi / Character LoRA Training
**Hedef / Goal**: 200-500 high-quality face images

```bash
python cli.py character_videos/*.mp4 \
  -f 1:1 \
  -i 20 \
  -c 0.7 \
  --ensemble \
  --turbo
```

**Output Structure:**
```
output/
└── video_name_1x1_ensemble_turbo/
    ├── persons/      # ✅ Your training images here!
    ├── animals/
    └── objects/
```

### Scenario 2: Hayvan/Pet LoRA / Animal/Pet LoRA
**Hedef / Goal**: Animal-specific dataset

```bash
python cli.py pet_videos/*.mp4 \
  -f 1:1 \
  -i 15 \
  -c 0.6 \
  --ensemble \
  --turbo
```

**Output**: Check `animals/` folder

### Scenario 3: Stil Transfer / Style Transfer
**Hedef / Goal**: Various scenes and objects

```bash
python cli.py style_videos/*.mp4 \
  -f 1:1 \
  -i 25 \
  -c 0.5 \
  --turbo
```

**Output**: All categories mixed

### Scenario 4: Hızlı Test / Quick Test
**Hedef / Goal**: Test before processing all videos

```bash
python cli.py test_video.mp4 \
  -f 1:1 \
  -i 90 \
  --no-turbo
```

**Time**: ~30 seconds for 10-minute video

---

## 🔥 İPUÇLARI / PRO TIPS

### 1. Video Hazırlama / Prepare Videos
```bash
# İyi video özellikleri / Good video qualities:
✅ 1080p or higher resolution
✅ Good lighting
✅ Minimal motion blur
✅ Diverse angles
✅ 5-10 minutes per video

# Kaçınılacaklar / Avoid:
❌ Low resolution (<720p)
❌ Heavy compression
❌ Extreme motion
❌ Poor lighting
```

### 2. Performans Optimizasyonu / Performance Optimization
```bash
# GPU kullanımı / GPU usage:
nvidia-smi  # Check GPU memory

# If memory error:
python cli.py videos/*.mp4 -i 60 --batch-size 2
```

### 3. Çıktı Kontrolü / Output Quality Control
```bash
# Çok az frame? / Too few frames?
python cli.py video.mp4 -i 15 -c 0.4  # Lower interval, lower confidence

# Çok fazla kötü frame? / Too many bad frames?
python cli.py video.mp4 -i 30 -c 0.7 --ensemble  # Higher confidence, use ensemble

# Yanlış tespitler? / False detections?
python cli.py video.mp4 --ensemble --voting-threshold 3  # Strict voting
```

### 4. Toplu İşlem Workflow / Batch Processing Workflow
```bash
# Step 1: Test with one video
python cli.py test.mp4 -f 1:1 -i 30

# Step 2: If good, process all
python cli.py videos/*.mp4 -f 1:1 -i 30 --ensemble --turbo

# Step 3: Check output folder
cd output
dir

# Step 4: Review and select best frames
```

---

## 📊 ÇIKTI ORGANİZASYONU / OUTPUT ORGANIZATION

### Klasör Yapısı / Folder Structure:
```
output/
├── video1_1x1_ensemble_turbo/
│   ├── persons/
│   │   ├── frame_000030_q85.jpg
│   │   ├── frame_000060_q92.jpg
│   │   └── ...
│   ├── animals/
│   └── objects/
├── video2_1x1_ensemble_turbo/
│   ├── persons/
│   ├── animals/
│   └── objects/
└── ...
```

### Dosya Adlandırma / File Naming:
- `frame_NNNNNN_qXX.jpg`
  - `NNNNNN`: Frame number (original video)
  - `qXX`: Quality score (30-100)
  - Higher quality = better centering, less cropping

---

## 🎓 SORU-CEVAP / FAQ

### Q: Hangi format LoRA için en iyi? / Which format is best for LoRA?
**A:** 1:1 (square) - Most LoRA models train on square images

### Q: Kaç frame gerekli? / How many frames needed?
**A:** 50-200 for basic, 200-500 for good quality, 500+ for excellent

### Q: Ensemble mode ne kadar yavaş? / How much slower is ensemble?
**A:** 2-3x slower but 10-20% more accurate

### Q: GPU gerekli mi? / Is GPU required?
**A:** No, but recommended (10x faster)

### Q: Toplu işlemde video sırası önemli mi? / Does video order matter in batch?
**A:** No, all videos processed sequentially

### Q: İşlem sırasında durdurabili miyim? / Can I stop during processing?
**A:** Yes, click "Stop" button (GUI) or Ctrl+C (CLI)

---

## ⚠️ SORUN GİDERME / TROUBLESHOOTING

### Error: Out of Memory
```bash
# Solution 1: Reduce batch size
python cli.py videos/*.mp4 --batch-size 2

# Solution 2: Increase frame interval
python cli.py videos/*.mp4 -i 60

# Solution 3: Disable ensemble
python cli.py videos/*.mp4 --no-turbo
```

### Error: No frames saved
```bash
# Solution: Lower confidence
python cli.py video.mp4 -c 0.3

# Or: Increase interval (process more frames)
python cli.py video.mp4 -i 15
```

### Error: Too many bad detections
```bash
# Solution: Use ensemble with strict voting
python cli.py video.mp4 --ensemble --voting-threshold 3 -c 0.7
```

---

## 🎯 HIZLI REFERANS / QUICK REFERENCE

### Hızlı Komutlar / Quick Commands:
```bash
# Test
python cli.py test.mp4 -i 90

# Normal
python cli.py video.mp4

# High Quality
python cli.py video.mp4 --ensemble -i 15 -c 0.7

# Batch Fast
python cli.py *.mp4 -i 50

# Batch Quality
python cli.py *.mp4 --ensemble -i 20
```

### Mod Kombinasyonları / Mode Combinations:
| Use Case | Command |
|----------|---------|
| En hızlı / Fastest | `-i 90 --no-turbo` |
| Hızlı / Fast | `-i 60 --turbo` |
| Dengeli / Balanced | `-i 30 --turbo` |
| Kaliteli / Quality | `-i 20 --ensemble --turbo` |
| En kaliteli / Best | `-i 10 --ensemble --voting-threshold 3` |

---

## 📚 İLERİ SEVİYE / ADVANCED

### Custom Pipeline:
```bash
# Step 1: Extract with low confidence (get everything)
python cli.py video.mp4 -c 0.3 -i 20 -o raw_output

# Step 2: Manual filtering
# Review and delete bad frames

# Step 3: Optional: Re-process with high quality
python cli.py selected_frames/*.jpg -c 0.8 --ensemble
```

### Automation Script (PowerShell):
```powershell
# Process all videos in subfolders
Get-ChildItem -Recurse -Include *.mp4 | ForEach-Object {
    python cli.py $_.FullName -f 1:1 -i 25 --ensemble --turbo
}
```

---

## 🎉 BAŞARILAR / SUCCESS TIPS

1. **Test etmeyi unutma** / **Always test first**
   - Bir video ile test et / Test with one video
   - Ayarları optimize et / Optimize settings
   - Sonra toplu işle / Then batch process

2. **Çeşitlilik önemli** / **Diversity matters**
   - Farklı açılar / Different angles
   - Farklı ışıklar / Different lighting
   - Farklı pozlar / Different poses

3. **Kalite kontrolü** / **Quality control**
   - Output'u gözden geçir / Review output
   - Kötü frame'leri sil / Delete bad frames
   - En iyileri seç / Select best ones

4. **İteratif süreç** / **Iterative process**
   - İlk sonuçları test et / Test first results
   - Ayarları fine-tune et / Fine-tune settings
   - Yeniden işle / Reprocess if needed

---

**🚀 Happy Harvesting! / İyi Hasatlar!**

For more details, see:
- `CHANGELOG.md` - Full changelog
- `README.md` - Complete documentation
- `ENSEMBLE.md` - Ensemble mode guide

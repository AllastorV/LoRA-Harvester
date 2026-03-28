# 🌾 LoRA-Harvester - Quick Reference Guide / Hızlı Referans Kılavuzu

[English](#english-guide) | [Türkçe](#turkce-kilavuz)

---

<a name="english-guide"></a>
## 🇬🇧 English Quick Guide

### 🚀 Quick Start
```bash
# Install
install.bat

# Run GUI
python main.py

# Run CLI
python cli.py your_video.mp4
```

### ⚡ Turbo Mode (2-3x Faster)
```bash
python cli.py video.mp4 --turbo --batch-size 8 -i 60
```

### 🤖 Ensemble Mode (Highest Accuracy)
```bash
python cli.py video.mp4 --ensemble --voting-threshold 2
```

### 📐 Output Formats
- `9:16`
- `3:4`
- `1:1`
- `4:5`
- `16:9`
- `4:3`

### 🎯 Important Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `-i, --interval` | Process every N frames | 30 |
| `-c, --confidence` | Detection confidence (0-1) | 0.5 |
| `-p, --padding` | Min padding in pixels | 500 |
| `-f, --format` | Aspect ratio | 9:16 |
| `--turbo` | Enable turbo mode | Off |
| `--ensemble` | Use 3 AI models | Off |

### 🖥️ GUI Features
- 🌐 Language: Turkish/English
- 🎨 Theme: Dark purple
- ❓ Tooltips: Hover over ? icons
- 🔄 Real-time progress
- ⏹️ Stop button

### 📝 Output Structure
```
output/video_name_9x16/
├── persons/    # Human detections
├── animals/    # Animal detections
└── objects/    # Other objects
```

### 🐛 Troubleshooting
- **Slow**: Use GPU, increase `-i`, enable `--turbo`
- **No output**: Lower `-c` to 0.4, try `--no-skip-text`
- **Memory error**: Reduce `--batch-size`, use smaller model

---

<a name="turkce-kilavuz"></a>
## 🇹🇷 Türkçe Hızlı Kılavuz

### 🚀 Hızlı Başlangıç
```bash
# Kur
install.bat

# GUI'yi Çalıştır
python main.py

# CLI'yi Çalıştır
python cli.py video_dosyaniz.mp4
```

### ⚡ Turbo Modu (2-3x Daha Hızlı)
```bash
python cli.py video.mp4 --turbo --batch-size 8 -i 60
```

### 🤖 Topluluk Modu (En Yüksek Doğruluk)
```bash
python cli.py video.mp4 --ensemble --voting-threshold 2
```

### 📐 Çıktı Formatları
- `9:16`
- `3:4`
- `1:1`
- `4:5`
- `16:9`
- `4:3`

### 🎯 Önemli Parametreler
| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `-i, --interval` | Her N karede işle | 30 |
| `-c, --confidence` | Tespit güveni (0-1) | 0.5 |
| `-p, --padding` | Min dolgu piksel | 500 |
| `-f, --format` | En-boy oranı | 9:16 |
| `--turbo` | Turbo modu aktif | Kapalı |
| `--ensemble` | 3 yapay zeka modeli kullan | Kapalı |

### 🖥️ GUI Özellikleri
- 🌐 Dil: Türkçe/İngilizce
- 🎨 Tema: Koyu mor
- ❓ İpuçları: ? simgelerinin üzerine gelin
- 🔄 Gerçek zamanlı ilerleme
- ⏹️ Durdurma butonu

### 📝 Çıktı Yapısı
```
output/video_adi_9x16/
├── persons/    # İnsan tespitleri
├── animals/    # Hayvan tespitleri
└── objects/    # Diğer nesneler
```

### 🐛 Sorun Giderme
- **Yavaş**: GPU kullan, `-i` artır, `--turbo` aç
- **Çıktı yok**: `-c` değerini 0.4'e düşür, `--no-skip-text` dene
- **Bellek hatası**: `--batch-size` azalt, küçük model kullan

---

## 📖 Full Documentation

For complete documentation, see **[README.md](README.md)**

Tam dokümantasyon için **[README.md](README.md)** dosyasına bakın

---

## 🎯 Example Commands / Örnek Komutlar

### Production Quality / Üretim Kalitesi
```bash
python cli.py video.mp4 -f 9:16 -i 25 -c 0.6 --turbo --ensemble
```

### Fast Test / Hızlı Test
```bash
python cli.py video.mp4 -i 90 --turbo --no-skip-text
```

### Maximum Accuracy / Maksimum Doğruluk
```bash
python cli.py video.mp4 -i 15 -c 0.7 -m yolov8m.pt --ensemble --voting-threshold 3
```

---

<div align="center">

**Video Smart Cropper** - AI-Powered Vertical Video Creator

**Yapay Zeka Destekli Dikey Video Oluşturucu**

</div>

# 🌾 LoRA-Harvester

<div align="center">

**AI-Powered Video Processing Tool for LoRA Training Dataset Creation**

**LoRA Eğitim Dataseti Oluşturma için Yapay Zeka Destekli Video İşleme Aracı**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-AllastorV-black.svg)](https://github.com/AllastorV)

*Accelerate LoRA training dataset collection with AI-powered smart cropping*

*LoRA eğitim dataseti toplama işlemini yapay zeka destekli akıllı kırpma ile hızlandırın*

**[GitHub Repository](https://github.com/AllastorV/LoRA-Harvester)**

[English](#english) | [Türkçe](#turkce)

</div>

---

<a name="english"></a>
## 🇬🇧 English Documentation

## 🎯 Purpose

This tool was created to **accelerate dataset collection for LoRA (Low-Rank Adaptation) training**. When training custom AI models with LoRA/Dreambooth, you need hundreds of quality images. Instead of manually extracting frames from videos, this tool:

✅ Automatically detects persons, animals, and objects  
✅ Intelligently crops to vertical formats  
✅ Skips low-quality frames and text overlays  
✅ Organizes output by category  
✅ Uses ensemble AI models for accuracy  

**Perfect for**: Character LoRA training, style transfer, object-specific model fine-tuning, and any AI training that requires consistent dataset creation from video sources.

---

## ✨ Features

### 🤖 AI-Powered Detection
- **YOLOv8 Integration**: State-of-the-art object detection
- **🆕 Ensemble Mode**: Combines 3 different AI models for higher accuracy
  - **YOLOv8** (Ultralytics) - Fast and accurate
  - **DETR** (Facebook/Meta) - Transformer-based detection
  - **Faster R-CNN** (Torchvision) - Traditional R-CNN architecture
- **Consensus Voting**: Multiple models verify each detection
- **Multi-Category Support**: Humans, animals, and objects
- **GPU Acceleration**: 10x faster processing with CUDA support
- **Confidence Threshold**: Adjustable detection sensitivity

### 🎨 Smart Cropping
- **Head Space Awareness**: Optimal framing for persons
- **Subject Centering**: Intelligent subject positioning
- **Adaptive Zoom**: Auto-adjusts crop size
- **Quality Scoring**: Only saves high-quality frames

### 📐 Multiple Formats
- **9:16**
- **3:4**
- **1:1**
- **4:5**
- **16:9**
- **4:3**

### 📝 Text Detection
- **Automatic Subtitle Skip**: Avoids text-heavy scenes
- **OCR Integration**: EasyOCR with multi-language support
- **Fast Mode**: Quick edge detection for performance

### 🖥️ User Interface
- **Modern PyQt5 GUI**: Intuitive drag & drop interface
- **Dark Purple Theme**: Easy on the eyes
- **Turkish/English**: Dual language support
- **Real-time Progress**: Live statistics and progress bar
- **CLI Mode**: Advanced command-line interface for automation
- **Tooltips**: Helpful hints for every setting

### 🗂️ Auto-Organization
Automatically creates organized output structure:
```
output/
└── video_name_9x16/
    ├── persons/      # Human detections - Perfect for character LoRA
    ├── animals/      # Animal detections - Animal LoRA training
    └── objects/      # Other objects - Object-specific training
```

---

## 🚀 Quick Start

### Option 1: Automatic Installation (Windows)
```bash
# Just double-click
install.bat

# Then run
run.bat
```

### Option 2: Manual Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GUI Mode
```bash
python main.py
```

### CLI Mode
```bash
# Basic usage
python cli.py video.mp4

# For LoRA training dataset (1:1 format, high quality)
python cli.py video.mp4 -f 1:1 -i 20 -c 0.7 --ensemble

# Fast processing
python cli.py video.mp4 -f 9:16 -i 60 --turbo
```

---

## 📖 Usage Examples

### Example 1: Character LoRA Dataset
```bash
# High quality, 1:1 format, ensemble mode for accuracy
python cli.py character_video.mp4 -f 1:1 -i 15 -c 0.7 --ensemble --turbo
```

### Example 2: Quick Dataset Collection
```bash
# Fast processing, good quality
python cli.py video.mp4 -f 1:1 -i 40 --turbo --batch-size 8
```

### Example 3: Animal LoRA Dataset
```bash
# Focus on animals, high precision
python cli.py pet_video.mp4 -f 1:1 -i 20 -c 0.6 --ensemble
```

### Example 4: Vertical Content (TikTok/Reels)
```bash
python cli.py content.mp4 -f 9:16 -i 30 -c 0.6 --turbo
```

### Example 5: Maximum Quality for Training
```bash
# All 3 models must agree, highest precision
python cli.py video.mp4 -f 1:1 -i 10 -c 0.8 --ensemble --voting-threshold 3
```

---

## 🎯 CLI Parameters

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| `video` | - | Input video file | Required |
| `--output` | `-o` | Output directory | `output` |
| `--format` | `-f` | Aspect ratio (9:16, 3:4, 1:1, 4:5, 16:9, 4:3) | `9:16` |
| `--interval` | `-i` | Frame interval | `30` |
| `--confidence` | `-c` | Detection confidence (0-1) | `0.5` |
| `--padding` | `-p` | Minimum padding (pixels) | `500` |
| `--model` | `-m` | YOLO model size | `yolov8n.pt` |
| `--ensemble` | - | Enable ensemble mode (3 models) | False |
| `--ensemble-models` | - | Models for ensemble | All 3 |
| `--voting-threshold` | - | Min model agreements | `2` |
| `--turbo` | - | Enable turbo mode (2-3x faster) | False |
| `--batch-size` | - | Batch size for turbo mode | `4` |
| `--no-skip-text` | - | Don't skip text frames | False |

---

## 🏗️ Project Structure

```
LoRA-Harvester/
├── src/
│   ├── core/
│   │   ├── detector.py             # YOLOv8 object detection
│   │   ├── ensemble_detector.py    # Multi-model ensemble
│   │   ├── text_detector.py        # Subtitle/text detection
│   │   ├── cropper.py              # Smart cropping algorithms
│   │   ├── video_processor.py      # Basic video processing
│   │   └── optimized_processor.py  # Turbo mode processing
│   ├── ui/
│   │   ├── main_window.py          # PyQt5 GUI
│   │   └── translations.py         # TR/EN translations
│   └── utils/
├── main.py                          # GUI entry point
├── cli.py                           # CLI entry point
├── config.yaml                      # Configuration file
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize default settings:

```yaml
detection:
  model_size: "yolov8n.pt"
  confidence: 0.5

cropping:
  default_format: "1:1"  # Best for LoRA training
  min_padding: 500

text_detection:
  enabled: true
  languages: ["en", "tr"]

ensemble:
  enabled: false
  voting_threshold: 2
```

---

## 🔧 Requirements

### System Requirements
- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 8GB+ (16GB recommended)
- **GPU**: CUDA-capable NVIDIA GPU (optional but recommended)
- **Storage**: Depends on video size

### Software Requirements
- Python 3.8 or higher
- CUDA Toolkit 11.8+ (for GPU acceleration)
- Tesseract OCR (for subtitle detection)

### Python Packages
All dependencies are listed in `requirements.txt`:
- PyTorch 2.0+
- Ultralytics (YOLOv8)
- Transformers (DETR)
- Torchvision (Faster R-CNN)
- OpenCV 4.8+
- PyQt5 5.15.9+
- EasyOCR 1.7+
- NumPy, Pillow

---

## 🎓 How It Works

1. **Frame Extraction**: Extracts frames at specified intervals
2. **Text Detection**: Skips frames with subtitles (optional)
3. **Object Detection**: 
   - **Single Mode**: YOLOv8 detects persons, animals, objects
   - **🆕 Ensemble Mode**: 3 models vote for consensus (higher accuracy)
4. **Smart Cropping**: 
   - Centers on primary subject
   - Adjusts for head space (persons)
   - Maintains aspect ratio
   - Adds padding
5. **Quality Check**: Evaluates and saves high-quality crops
6. **Auto-Organization**: Sorts into person/animal/object folders

### 🤖 Ensemble Detection Workflow

```
Frame → ┌─ YOLOv8 (Fast)        → person, dog, car
        ├─ DETR (Transformer)    → person, dog
        └─ Faster R-CNN (Precise) → person, dog, tree
                ↓
        Voting (2/3 threshold)
                ↓
        Consensus: person ✅, dog ✅
        (car ❌, tree ❌ - not enough votes)
```

---

## 🚀 Performance Tips

### For LoRA Dataset Creation
```bash
# Balanced: Good quality, reasonable speed
python cli.py video.mp4 -f 1:1 -i 25 -c 0.6 --turbo

# High quality: Best for training
python cli.py video.mp4 -f 1:1 -i 15 -c 0.7 --ensemble --turbo

# Fast collection: Quick iteration
python cli.py video.mp4 -f 1:1 -i 60 --turbo --batch-size 8
```

### GPU vs CPU
- **GPU**: ~10x faster, recommended for ensemble mode
- **CPU**: Works but slower, increase frame interval

### Model Selection
- `yolov8n.pt`: Fast, good accuracy ✅ (recommended)
- `yolov8s.pt`: Balanced
- `yolov8m.pt`: More accurate, slower
- `yolov8l.pt`: Most accurate, very slow

---

## 🐛 Troubleshooting

### CUDA Not Available
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory
- Use smaller model (`yolov8n.pt`)
- Increase frame interval
- Reduce batch size (`--batch-size 2`)

### No Frames Saved
- Lower confidence threshold (`-c 0.4`)
- Decrease frame interval (`-i 15`)
- Try `--no-skip-text`

### Slow Processing
- Use GPU
- Increase frame interval (`-i 90`)
- Enable turbo mode (`--turbo`)
- Use smaller model

For detailed guides:
- [QUICKSTART.md](QUICKSTART.md) - Quick reference
- [ENSEMBLE.md](ENSEMBLE.md) - Ensemble mode detailed guide
- [OPTIMIZATION.md](OPTIMIZATION.md) - Performance optimization

---

## 🎬 Use Cases

- **LoRA Training**: Rapid dataset collection for Stable Diffusion LoRA
- **Dreambooth**: Character/object dataset preparation
- **Content Creators**: Repurpose YouTube videos for TikTok/Reels
- **Marketing Teams**: Create vertical format ads from landscape footage
- **AI Model Training**: Consistent dataset creation from video sources

---

## 📝 License

GNU General Public License v3.0 - free for personal and commercial use with copyleft requirements

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues and pull requests on [GitHub](https://github.com/AllastorV/LoRA-Harvester).

---

## 📧 Support

For issues and questions:
1. Check documentation files
2. Review troubleshooting section
3. Open an issue on [GitHub](https://github.com/AllastorV/LoRA-Harvester/issues)

---

## 🌟 Acknowledgments

- **YOLOv8** by Ultralytics
- **DETR** by Facebook/Meta
- **Faster R-CNN** by Torchvision
- **EasyOCR** for text detection
- **PyTorch** for deep learning
- **OpenCV** for video processing

---

<a name="turkce"></a>
## 🇹🇷 Türkçe Dokümantasyon

## 🎯 Amaç

Bu araç, **LoRA (Low-Rank Adaptation) eğitimi için dataset toplama işlemini hızlandırmak** amacıyla oluşturulmuştur. LoRA/Dreambooth ile özel AI modelleri eğitirken yüzlerce kaliteli görüntüye ihtiyacınız vardır. Videolardan manuel olarak kare çıkarmak yerine, bu araç:

✅ Kişileri, hayvanları ve nesneleri otomatik tespit eder  
✅ Akıllıca dikey formatlara kırpar  
✅ Düşük kaliteli kareleri ve metin yerleşimlerini atlar  
✅ Çıktıyı kategorilere göre düzenler  
✅ Doğruluk için ensemble AI modelleri kullanır  

**Mükemmel kullanım alanları**: Karakter LoRA eğitimi, stil transferi, nesne-özel model ince ayarı ve video kaynaklarından tutarlı dataset oluşturmayı gerektiren her AI eğitimi.

---

## ✨ Özellikler

### 🤖 Yapay Zeka Destekli Tespit
- **YOLOv8 Entegrasyonu**: Son teknoloji nesne tespiti
- **🆕 Topluluk Modu**: Daha yüksek doğruluk için 3 farklı yapay zeka modeli
  - **YOLOv8** (Ultralytics) - Hızlı ve doğru
  - **DETR** (Facebook/Meta) - Transformer tabanlı tespit
  - **Faster R-CNN** (Torchvision) - Geleneksel R-CNN mimarisi
- **Konsensüs Oylama**: Birden fazla model her tespiti doğrular
- **Çoklu Kategori Desteği**: İnsanlar, hayvanlar ve nesneler
- **GPU Hızlandırma**: CUDA desteği ile 10x daha hızlı işleme
- **Güven Eşiği**: Ayarlanabilir tespit hassasiyeti

### 🎨 Akıllı Kırpma
- **Baş Boşluğu Farkındalığı**: Kişiler için optimal çerçeveleme
- **Özne Merkezleme**: Akıllı özne konumlandırma
- **Uyarlamalı Zoom**: Kırpma boyutunu otomatik ayarlar
- **Kalite Puanlama**: Sadece yüksek kaliteli kareleri kaydeder

### 📐 Çoklu Formatlar
- **9:16** 
- **3:4**
- **1:1**
- **4:5**
- **16:9**
- **4:3**

### 📝 Metin Tespiti
- **Otomatik Altyazı Atlama**: Metin yoğun sahnelerden kaçınır
- **OCR Entegrasyonu**: Çok dilli EasyOCR desteği
- **Hızlı Mod**: Performans için hızlı kenar tespiti

### 🖥️ Kullanıcı Arayüzü
- **Modern PyQt5 GUI**: Sezgisel sürükle-bırak arayüzü
- **Karanlık Mor Tema**: Göz yormayan tasarım
- **Türkçe/İngilizce**: Çift dil desteği
- **Gerçek Zamanlı İlerleme**: Canlı istatistikler ve ilerleme çubuğu
- **CLI Modu**: Otomasyon için gelişmiş komut satırı arayüzü
- **Tooltip Yardım**: Her ayar için açıklayıcı ipuçları

### 🗂️ Otomatik Organizasyon
Otomatik olarak düzenli çıktı yapısı oluşturur:
```
output/
└── video_adi_9x16/
    ├── persons/      # İnsan tespitleri - Karakter LoRA için mükemmel
    ├── animals/      # Hayvan tespitleri - Hayvan LoRA eğitimi
    └── objects/      # Diğer nesneler - Nesne-özel eğitim
```

---

## 🚀 Hızlı Başlangıç

### Seçenek 1: Otomatik Kurulum (Windows)
```bash
# Sadece çift tıklayın
install.bat

# Sonra çalıştırın
run.bat
```

### Seçenek 2: Manuel Kurulum
```bash
# Sanal ortam oluştur
python -m venv venv

# Aktif et (Windows)
venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### GUI Modu
```bash
python main.py
```

### CLI Modu
```bash
# Basit kullanım
python cli.py video.mp4

# LoRA eğitim dataseti için (1:1 format, yüksek kalite)
python cli.py video.mp4 -f 1:1 -i 20 -c 0.7 --ensemble

# Hızlı işleme
python cli.py video.mp4 -f 9:16 -i 60 --turbo
```

---

## 📖 Kullanım Örnekleri

### Örnek 1: Karakter LoRA Dataseti
```bash
# Yüksek kalite, 1:1 format, doğruluk için ensemble modu
python cli.py karakter_video.mp4 -f 1:1 -i 15 -c 0.7 --ensemble --turbo
```

### Örnek 2: Hızlı Dataset Toplama
```bash
# Hızlı işleme, iyi kalite
python cli.py video.mp4 -f 1:1 -i 40 --turbo --batch-size 8
```

### Örnek 3: Hayvan LoRA Dataseti
```bash
# Hayvanlara odaklan, yüksek hassasiyet
python cli.py evcil_hayvan_video.mp4 -f 1:1 -i 20 -c 0.6 --ensemble
```

### Örnek 4: Dikey İçerik (TikTok/Reels)
```bash
python cli.py icerik.mp4 -f 9:16 -i 30 -c 0.6 --turbo
```

### Örnek 5: Eğitim İçin Maksimum Kalite
```bash
# 3 modelin de anlaşması gerekli, en yüksek hassasiyet
python cli.py video.mp4 -f 1:1 -i 10 -c 0.8 --ensemble --voting-threshold 3
```

---

## 🎯 CLI Parametreleri

| Parametre | Kısa | Açıklama | Varsayılan |
|-----------|------|----------|------------|
| `video` | - | Giriş video dosyası | Zorunlu |
| `--output` | `-o` | Çıktı dizini | `output` |
| `--format` | `-f` | En-boy oranı (9:16, 3:4, 1:1, 4:5, 16:9, 4:3) | `9:16` |
| `--interval` | `-i` | Kare aralığı | `30` |
| `--confidence` | `-c` | Tespit güveni (0-1) | `0.5` |
| `--padding` | `-p` | Minimum dolgu (piksel) | `500` |
| `--model` | `-m` | YOLO model boyutu | `yolov8n.pt` |
| `--ensemble` | - | Topluluk modunu aktifleştir | False |
| `--ensemble-models` | - | Topluluk için modeller | 3'ü de |
| `--voting-threshold` | - | Min model anlaşması | `2` |
| `--turbo` | - | Turbo modu aktif (2-3x hızlı) | False |
| `--batch-size` | - | Turbo modu toplu boyutu | `4` |
| `--no-skip-text` | - | Metin karelerini atlama | False |

---

## 🚀 LoRA Dataset Oluşturma İpuçları

### LoRA Eğitimi İçin
```bash
# Dengeli: İyi kalite, makul hız
python cli.py video.mp4 -f 1:1 -i 25 -c 0.6 --turbo

# Yüksek kalite: Eğitim için en iyisi
python cli.py video.mp4 -f 1:1 -i 15 -c 0.7 --ensemble --turbo

# Hızlı toplama: Hızlı iterasyon
python cli.py video.mp4 -f 1:1 -i 60 --turbo --batch-size 8
```

### GPU vs CPU
- **GPU**: ~10x daha hızlı, ensemble modu için önerilir
- **CPU**: Çalışır ama daha yavaş, kare aralığını artırın

---

## 🎬 Kullanım Alanları

- **LoRA Eğitimi**: Stable Diffusion LoRA için hızlı dataset toplama
- **Dreambooth**: Karakter/nesne dataset hazırlama
- **İçerik Üreticileri**: YouTube videolarını TikTok/Reels için yeniden kullanın
- **Pazarlama Ekipleri**: Yatay çekimlerden dikey format reklamlar oluşturun
- **AI Model Eğitimi**: Video kaynaklarından tutarlı dataset oluşturma

---

## 📝 Lisans

GNU Genel Kamu Lisansı v3.0 - copyleft gereklilikleri ile kişisel ve ticari kullanım için ücretsiz

Bu program özgür bir yazılımdır: Free Software Foundation tarafından yayınlanan GNU Genel Kamu Lisansı'nın 3. sürümü şartları altında yeniden dağıtabilir ve/veya değiştirebilirsiniz.

Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen [GitHub](https://github.com/AllastorV/LoRA-Harvester)'da sorunları ve pull request'leri göndermekten çekinmeyin.

---

## 📧 Destek

Sorunlar ve sorular için:
1. Dokümantasyon dosyalarını kontrol edin
2. Sorun giderme bölümünü inceleyin
3. [GitHub](https://github.com/AllastorV/LoRA-Harvester/issues)'da issue açın

---

## 🌟 Teşekkürler

- **YOLOv8** by Ultralytics
- **DETR** by Facebook/Meta
- **Faster R-CNN** by Torchvision
- **EasyOCR** metin tespiti için
- **PyTorch** derin öğrenme için
- **OpenCV** video işleme için

---

<div align="center">

**❤️ ile yapıldı - Yapay Zeka ve GPU hızlandırma kullanılarak**

**Made with ❤️ using AI and GPU acceleration**

**[GitHub @AllastorV](https://github.com/AllastorV)**

*Faydalı bulursanız ⭐ yıldız verin! / Star ⭐ if you find it useful!*

</div>




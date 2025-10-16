# 🌾 LoRA-Harvester

<div align="center">


### 🎯 AI-Powered Video Processing Tool for LoRA Training Dataset Creation
 🎯 LoRA Eğitim Dataseti Oluşturma için Yapay Zeka Destekli Video İşleme Aracı

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue?style=flat-square&logo=gnu&logoColor=white)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-AllastorV-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/AllastorV)

<img src="https://img.shields.io/badge/AI-Powered-blueviolet?style=for-the-badge&logo=opencv&logoColor=white" alt="AI Powered">
<img src="https://img.shields.io/badge/LoRA-Training-ff69b4?style=for-the-badge&logo=pytorch&logoColor=white" alt="LoRA Training">
<img src="https://img.shields.io/badge/Batch-Processing-success?style=for-the-badge&logo=files&logoColor=white" alt="Batch Processing">




**⚡ Accelerate LoRA training dataset collection with AI-powered smart cropping**

⚡ LoRA eğitim dataseti toplama işlemini yapay zeka destekli akıllı kırpma ile hızlandırın**

---



**[📖 Documentation](#english) | [📖 Dokümantasyon](#turkce)**

</div>

---

<a name="english"></a>

<div align="center">

## 🇬🇧 ENGLISH DOCUMENTATION

<img src="https://img.shields.io/badge/Language-English-blue?style=for-the-badge" alt="English">

</div>

---

## 🎯 Purpose & Vision

<table>
<tr>
<td width="50%">

### 🎨 **What is LoRA-Harvester?**

A powerful AI-driven tool designed to **revolutionize dataset creation** for LoRA/Dreambooth training. Transform hours of manual work into minutes of automated processing.

</td>
<td width="50%">

### 🚀 **Why Use It?**

Instead of manually extracting hundreds of frames, let AI do the heavy lifting:
- ✅ **10x faster** than manual extraction
- ✅ **Higher quality** with smart detection
- ✅ **Organized output** ready for training

</td>
</tr>
</table>

### 🎬 Perfect For:
```
┌─────────────────────┬──────────────────────┬─────────────────────┐
│  👤 Character LoRA  │  🐾 Animal/Pet LoRA  │  🎨 Style Transfer │
│  Face training      │  Pet recognition     │  Artistic styles    │
│  Portrait datasets  │  Wildlife datasets   │  Object datasets    │
└─────────────────────┴──────────────────────┴─────────────────────┘
```

---

## ✨ Key Features

<div align="center">

### 🤖 **AI-Powered Detection**

</div>

<table>
<tr>
<td width="33%" align="center">

#### 🎯 YOLOv8
**Fast & Accurate**
- State-of-the-art detection
- Real-time processing
- GPU accelerated

</td>
<td width="33%" align="center">

#### 🧠 Ensemble Mode
**3 AI Models**
- YOLO + DETR + Faster R-CNN
- Voting mechanism
- 95%+ accuracy

</td>
<td width="33%" align="center">

#### ⚡ Turbo Mode
**2-3x Faster**
- Batch processing
- FP16 inference
- Optimized memory

</td>
</tr>
</table>

<div align="center">

### 🎨 **Smart Processing**

</div>

| Feature | Description | Benefit |
|---------|-------------|---------|
| **📐 Multiple Formats** | 9:16, 3:4, 1:1, 4:5, 16:9, 4:3 | Perfect crop for any use case |
| **🎯 Smart Cropping** | Head space awareness + centering | Professional-quality framing |
| **📝 Text Detection** | Auto-skip subtitles | Clean, text-free images |
| **💎 Quality Scoring** | Automatic quality assessment | Only save the best frames |
| **🗂️ Auto-Organization** | Categorized by persons/animals/objects | Training-ready structure |

<div align="center">

### 🚀 **NEW: Batch Processing**

</div>

```
┌──────────────────────────────────────────────────────────────┐
│  📹 Video 1  →  ✅ Processed  →  💾 150 frames sav         │
│  📹 Video 2  →  ✅ Processed  →  💾 200 frames saved       │
│  📹 Video 3  →  ✅ Processed  →  💾 180 frames saved       │
│                                                              │
│  ✅ TOTAL: 3 videos, 530 frames in 5 minutes!               │
└──────────────────────────────────────────────────────────────┘
```

**🎉 Process unlimited videos in one command!**

---

## 🚀 Quick Start

<details open>
<summary><b>📦 Option 1: Automatic Installation (Windows - Recommended)</b></summary>

```bash
# Just double-click these files:
install.bat          # Install everything automatically
run.bat             # Launch GUI mode
run_batch.bat       # Launch batch processing wizard
```

</details>

<details>
<summary><b>🔧 Option 2: Manual Installation</b></summary>

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run!
python main.py
```

</details>

<details>
<summary><b>🐍 Option 3: Direct Python</b></summary>

```bash
pip install -r requirements.txt
python main.py
```

</details>

---

## 💡 Usage Examples

### 🖥️ **GUI Mode** (Beginner-Friendly)

<table>
<tr>
<td width="50%">

#### Single Video
1. Launch: `python main.py`
2. Drag & drop video
3. Adjust settings
4. Click **Start Processing**

</td>
<td width="50%">

#### Batch Processing (NEW!)
1. Launch: `python main.py`
2. Drag & drop **multiple videos**
3. Or use Browse (Ctrl+Click)
4. Click **Start Processing**

</td>
</tr>
</table>

### ⌨️ **CLI Mode** (Advanced Users)

<details open>
<summary><b>🎯 Single Video Processing</b></summary>

```bash
# Basic usage
python cli.py video.mp4

# Custom settings
python cli.py video.mp4 -f 1:1 -i 30 -c 0.6

# High quality mode
python cli.py video.mp4 -f 1:1 -i 15 -c 0.7 --ensemble --turbo
```

</details>

<details open>
<summary><b>📹 Batch Processing (NEW!)</b></summary>

```bash
# Method 1: List videos
python cli.py video1.mp4 video2.mp4 video3.mp4

# Method 2: Use wildcards
python cli.py *.mp4
python cli.py videos/*.mp4

# Method 3: Batch wizard
run_batch.bat

# Method 4: High quality batch
python cli.py videos/*.mp4 -f 1:1 -i 20 --ensemble --turbo
```

</details>

<details>
<summary><b>🎨 Real-World Examples</b></summary>

#### Example 1: Character LoRA Training Dataset
```bash
python cli.py character_video.mp4 -f 1:1 -i 15 -c 0.7 --ensemble --turbo
```
**Result**: 200-500 high-quality face crops

#### Example 2: Pet/Animal LoRA Dataset
```bash
python cli.py pet_videos/*.mp4 -f 1:1 -i 20 -c 0.6 --ensemble
```
**Result**: Consistent animal photos from multiple angles

#### Example 3: Vertical Content (TikTok/Reels)
```bash
python cli.py content.mp4 -f 9:16 -i 30 --turbo
```
**Result**: Vertical format crops ready for social media

#### Example 4: Maximum Quality (Strict Mode)
```bash
python cli.py video.mp4 -f 1:1 -i 10 -c 0.8 --ensemble --voting-threshold 3
```
**Result**: Only frames where all 3 AI models agree

#### Example 5: Fast Preview
```bash
python cli.py test.mp4 -f 1:1 -i 90
```
**Result**: Quick test in 30 seconds

</details>

---

## 🎛️ Parameters & Settings

<div align="center">

### 📋 **Complete Parameter Reference**

</div>

| Parameter | Short | Options | Default | Description |
|-----------|-------|---------|---------|-------------|
| `videos` | - | file paths | *required* | 🎬 Single or multiple video files |
| `--output` | `-o` | path | `output` | 📁 Output directory |
| `--format` | `-f` | 9:16, 3:4, 1:1, 4:5, 16:9, 4:3 | `9:16` | 📐 Aspect ratio |
| `--interval` | `-i` | 1-200 | `30` | ⏱️ Process every N frames |
| `--confidence` | `-c` | 0.1-0.95 | `0.5` | 🎯 Detection threshold |
| `--padding` | `-p` | 100-1000 | `500` | 📏 Min padding (pixels) |
| `--model` | `-m` | n/s/m/l | `yolov8n.pt` | 🤖 YOLO model size |
| `--ensemble` | - | flag | OFF | 🧠 Enable 3-model ensemble |
| `--ensemble-models` | - | yolo, detr, fasterrcnn | all 3 | 🎯 Models for ensemble |
| `--voting-threshold` | - | 1-3 | `2` | 🗳️ Min model agreements |
| `--turbo` | - | flag | **ON** | ⚡ Batch frame processing |
| `--no-turbo` | - | flag | OFF | 🐌 Disable turbo mode |
| `--batch-size` | - | 1-16 | `4` | 📦 Frames per batch |
| `--no-skip-text` | - | flag | OFF | 📝 Process text frames |

<div align="center">

### 🎨 **Recommended Presets**

</div>

<table>
<tr>
<th>Use Case</th>
<th>Command</th>
<th>Speed</th>
<th>Quality</th>
</tr>
<tr>
<td>🏆 <b>LoRA Training (Best)</b></td>
<td><code>-f 1:1 -i 15 -c 0.7 --ensemble --turbo</code></td>
<td>⚡⚡</td>
<td>⭐⭐⭐⭐⭐</td>
</tr>
<tr>
<td>⚡ <b>Fast Collection</b></td>
<td><code>-f 1:1 -i 50 --turbo</code></td>
<td>⚡⚡⚡⚡</td>
<td>⭐⭐⭐</td>
</tr>
<tr>
<td>📱 <b>Vertical Content</b></td>
<td><code>-f 9:16 -i 30 --turbo</code></td>
<td>⚡⚡⚡</td>
<td>⭐⭐⭐⭐</td>
</tr>
<tr>
<td>💎 <b>Maximum Quality</b></td>
<td><code>-f 1:1 -i 10 --ensemble --voting-threshold 3</code></td>
<td>⚡</td>
<td>⭐⭐⭐⭐⭐</td>
</tr>
<tr>
<td>🧪 <b>Quick Test</b></td>
<td><code>-f 1:1 -i 90</code></td>
<td>⚡⚡⚡⚡⚡</td>
<td>⭐⭐</td>
</tr>
</table>

---

## 🏗️ Project Architecture

<div align="center">

### 📁 **File Structure**

</div>

```
🌾 LoRA-Harvester/
│
├── 🚀 Entry Points
│   ├── main.py                      # GUI launcher
│   ├── cli.py                       # CLI interface (batch support)
│   ├── run.bat                      # Quick start script
│   └── run_batch.bat                # Batch processing wizard
│
├── 🧠 Core Engine (src/core/)
│   ├── unified_processor.py         # ⭐ All-in-one processor
│   ├── detector.py                  # YOLOv8 detection
│   ├── ensemble_detector.py         # Multi-model ensemble
│   ├── text_detector.py             # Subtitle detection
│   └── cropper.py                   # Smart cropping
│
├── 🎨 User Interface (src/ui/)
│   ├── main_window.py               # PyQt5 GUI (batch support)
│   └── translations.py              # TR/EN translations
│
├── 📚 Documentation
│   ├── README.md                    # This file
│   ├── CHANGELOG.md                 # What's new
│   ├── QUICKSTART_NEW.md           # Quick reference
│   ├── ENSEMBLE.md                  # Ensemble guide
│   └── OPTIMIZATION.md              # Performance tips
│
└── ⚙️ Configuration
    ├── config.yaml                  # Settings
    ├── requirements.txt             # Dependencies
    └── yolov8n.pt                  # AI model
```

<div align="center">

### 🔄 **Processing Pipeline**

</div>

```
┌─────────────┐
│ 📹 Video(s) │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ 🤖 AI Detection     │──┐
│ • YOLOv8            │  │ Ensemble Mode
│ • DETR (optional)   │◄─┤ (3 models vote)
│ • Faster R-CNN      │  │
└──────┬──────────────┘──┘
       │
       ▼
┌─────────────────────┐
│ 📝 Text Detection   │
│ Skip subtitles?     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ ✂️ Smart Cropping   │
│ • Head space calc   │
│ • Subject centering │
│ • Format adjustment │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 💎 Quality Check    │
│ Score: 0.0 - 1.0    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 💾 Save & Organize  │
│ • persons/          │
│ • animals/          │
│ • objects/          │
└─────────────────────┘
```

---

## ⚙️ Configuration

<details>
<summary><b>📝 config.yaml Settings</b></summary>

```yaml
# Detection Settings
detection:
  model_size: "yolov8n.pt"        # n=fast, s=balanced, m/l=accurate
  confidence: 0.5                  # 0.1-0.95
  
# Cropping Settings
cropping:
  default_format: "1:1"            # Best for LoRA training
  min_padding: 500                 # Pixels around subject
  
# Text Detection
text_detection:
  enabled: true                    # Skip subtitle frames
  languages: ["en", "tr"]          # Supported languages
  
# Ensemble Mode
ensemble:
  enabled: false                   # Enable in CLI with --ensemble
  voting_threshold: 2              # Min agreements (1-3)
  models: ["yolo", "detr", "fasterrcnn"]
  
# Performance
performance:
  turbo_mode: true                 # Batch processing
  batch_size: 4                    # Frames per batch
  use_fp16: true                   # Half precision (if supported)
```

</details>

---

## 🔧 System Requirements

<div align="center">

### 💻 **Hardware Requirements**

</div>

<table>
<tr>
<th></th>
<th>Minimum</th>
<th>Recommended</th>
</tr>
<tr>
<td><b>CPU</b></td>
<td>Intel i5 / AMD Ryzen 5</td>
<td>Intel i7 / AMD Ryzen 7</td>
</tr>
<tr>
<td><b>RAM</b></td>
<td>8 GB</td>
<td>16 GB</td>
</tr>
<tr>
<td><b>GPU</b></td>
<td>Optional (CPU mode)</td>
<td>NVIDIA GTX 1060 6GB</td>
</tr>
<tr>
<td><b>Storage</b></td>
<td>10 GB free</td>
<td>20 GB free</td>
</tr>
</table>

<div align="center">

### 📦 **Software Requirements**

</div>

| Software | Version | Required | Notes |
|----------|---------|----------|-------|
| **Python** | 3.8 - 3.11 | ✅ Yes | Python 3.12 not yet supported |
| **CUDA Toolkit** | 11.8+ | ⚠️ GPU only | For NVIDIA GPU acceleration |
| **Tesseract OCR** | Latest | ⚠️ Optional | For advanced text detection |
| **Windows** | 10/11 | ✅ Recommended | Linux/Mac also supported |

---

## 🚀 Performance Guide

<div align="center">

### ⚡ **Speed Comparison**

</div>

| Mode | GPU | CPU | 10min Video |
|------|-----|-----|-------------|
| **Standard (YOLO)** | ~30 FPS | ~5 FPS | 20-30 sec |
| **Turbo (YOLO)** | ~60 FPS | ~10 FPS | 10-15 sec |
| **Ensemble (3 models)** | ~10 FPS | ~2 FPS | 60-90 sec |
| **Ensemble + Turbo** | ~20 FPS | ~4 FPS | 30-45 sec |

<div align="center">

### 💡 **Optimization Tips**

</div>

<table>
<tr>
<td width="33%">

#### 🐌 Too Slow?
- ✅ Enable turbo mode
- ✅ Increase frame interval
- ✅ Use smaller YOLO model
- ✅ Disable ensemble mode

</td>
<td width="33%">

#### 💾 Out of Memory?
- ✅ Reduce batch size
- ✅ Increase frame interval
- ✅ Use CPU mode
- ✅ Close other programs

</td>
<td width="33%">

#### 📉 Poor Quality?
- ✅ Enable ensemble mode
- ✅ Decrease frame interval
- ✅ Increase confidence
- ✅ Use larger YOLO model

</td>
</tr>
</table>

---

## 🎬 Use Cases & Applications

<div align="center">

### 🎯 **Real-World Applications**

</div>

<table>
<tr>
<td align="center" width="25%">

### 👤 Character LoRA
**Stable Diffusion**

Train custom character models

✅ Face consistency  
✅ Multiple angles  
✅ Various expressions  

</td>
<td align="center" width="25%">

### 🐾 Animal/Pet LoRA
**Pet Recognition**

Create pet-specific models

✅ Pet portraits  
✅ Breed training  
✅ Wildlife datasets  

</td>
<td align="center" width="25%">

### 🎨 Style Transfer
**Artistic AI**

Train style models

✅ Art styles  
✅ Object consistency  
✅ Scene datasets  



</td>
</tr>
</table>

---

## 🐛 Troubleshooting

<details>
<summary><b>❌ CUDA Not Available</b></summary>

**Problem**: Running on CPU, slow performance

**Solution**:
```bash
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

</details>

<details>
<summary><b>💾 Out of Memory Error</b></summary>

**Problem**: GPU/RAM out of memory

**Solutions**:
```bash
# Option 1: Reduce batch size
python cli.py video.mp4 --batch-size 2

# Option 2: Increase frame interval
python cli.py video.mp4 -i 60

# Option 3: Use smaller model
python cli.py video.mp4 -m yolov8n.pt

# Option 4: Disable ensemble
python cli.py video.mp4  # No --ensemble flag
```

</details>

<details>
<summary><b>📉 No Frames Saved / Too Few Frames</b></summary>

**Problem**: Output folder empty or very few frames

**Solutions**:
```bash
# Option 1: Lower confidence threshold
python cli.py video.mp4 -c 0.3

# Option 2: Process more frames
python cli.py video.mp4 -i 15

# Option 3: Include text frames
python cli.py video.mp4 --no-skip-text

# Option 4: Check video quality
# Make sure video is clear and well-lit
```

</details>

<details>
<summary><b>🤖 Ensemble Mode Too Slow</b></summary>

**Problem**: Ensemble takes too long

**Solutions**:
```bash
# Option 1: Use only 2 models
python cli.py video.mp4 --ensemble --ensemble-models yolo detr

# Option 2: Increase frame interval
python cli.py video.mp4 --ensemble -i 60

# Option 3: Enable turbo (if not already)
python cli.py video.mp4 --ensemble --turbo

# Option 4: Use GPU
# Ensemble mode really needs GPU
```

</details>

<details>
<summary><b>📝 Too Many False Positives</b></summary>

**Problem**: Saving bad/incorrect detections

**Solutions**:
```bash
# Option 1: Use ensemble with strict voting
python cli.py video.mp4 --ensemble --voting-threshold 3

# Option 2: Increase confidence
python cli.py video.mp4 -c 0.7

# Option 3: Combine both
python cli.py video.mp4 --ensemble --voting-threshold 3 -c 0.7
```

</details>

---

## 📚 Additional Resources

<div align="center">

### 📖 **Documentation**

</div>

| Document | Description | Link |
|----------|-------------|------|
| 🚀 **QUICKSTART_NEW.md** | Quick reference & examples | [View](QUICKSTART_NEW.md) |
| 📝 **CHANGELOG.md** | What's new in latest version | [View](CHANGELOG.md) |
| 🤖 **ENSEMBLE.md** | Ensemble mode detailed guide | [View](ENSEMBLE.md) |
| ⚡ **OPTIMIZATION.md** | Performance optimization tips | [View](OPTIMIZATION.md) |

<div align="center">

### 🔗 **External Links**

</div>

- 🐙 **GitHub Repository**: [AllastorV/LoRA-Harvester](https://github.com/AllastorV/LoRA-Harvester)
- 📧 **Issues & Support**: [GitHub Issues](https://github.com/AllastorV/LoRA-Harvester/issues)
- 🌟 **Star on GitHub**: [Give us a ⭐](https://github.com/AllastorV/LoRA-Harvester)

---

## 📝 License & Credits

<div align="center">

### 📜 **License**

**GNU General Public License v3.0**

Free for personal and commercial use with copyleft requirements

</div>

```
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.
```

<div align="center">

### 🙏 **Acknowledgments**

</div>

<table>
<tr>
<td align="center">
<b>🤖 YOLOv8</b><br>
by Ultralytics<br>
Object Detection
</td>
<td align="center">
<b>🧠 DETR</b><br>
by Facebook/Meta<br>
Transformer Detection
</td>
<td align="center">
<b>🔍 Faster R-CNN</b><br>
by Torchvision<br>
Region-based Detection
</td>
<td align="center">
<b>📝 EasyOCR</b><br>
by JaidedAI<br>
Text Detection
</td>
</tr>
<tr>
<td align="center">
<b>🔥 PyTorch</b><br>
Deep Learning<br>
Framework
</td>
<td align="center">
<b>🎨 OpenCV</b><br>
Video Processing<br>
Library
</td>
<td align="center">
<b>🖼️ PyQt5</b><br>
GUI Framework<br>
Interface
</td>
<td align="center">
<b>💜 Community</b><br>
Contributors<br>
& Users
</td>
</tr>
</table>

---

## 🤝 Contributing

<div align="center">

**🎉 Contributions are welcome!**

</div>

1. 🍴 Fork the repository
2. 🌿 Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🎁 Open a Pull Request

---

## 📧 Support

<div align="center">

### 💬 **Need Help?**

</div>

<table>
<tr>
<td align="center" width="33%">

### 📖 Documentation
Check our guides first

[Read Docs](#documentation)

</td>
<td align="center" width="33%">

### 🐛 Bug Report
Found an issue?

[Report on GitHub](https://github.com/AllastorV/LoRA-Harvester/issues)

</td>
<td align="center" width="33%">

### 💡 Feature Request
Have an idea?

[Suggest on GitHub](https://github.com/AllastorV/LoRA-Harvester/issues)

</td>
</tr>
</table>

---

<a name="turkce"></a>

<div align="center">

## 🇹🇷 TÜRKÇE DOKÜMANTASYON

<img src="https://img.shields.io/badge/Dil-Türkçe-red?style=for-the-badge" alt="Türkçe">

</div>

---

## 🎯 Amaç & Vizyon

<table>
<tr>
<td width="50%">

### 🎨 **LoRA-Harvester Nedir?**

LoRA/Dreambooth eğitimi için **dataset oluşturmayı devrimleştiren** güçlü bir yapay zeka aracı. Saatlerce süren manuel işi dakikalara indirin.

</td>
<td width="50%">

### 🚀 **Neden Kullanmalı?**

Yüzlerce kareyi elle çıkarmak yerine, yapay zekanın işi yapmasına izin verin:
- ✅ Manuel çıkarmadan **10x daha hızlı**
- ✅ Akıllı tespit ile **daha yüksek kalite**
- ✅ Eğitime hazır **organize çıktı**

</td>
</tr>
</table>

### 🎬 Mükemmel Kullanım Alanları:
```
┌─────────────────────┬──────────────────────┬─────────────────────┐
│  👤 Karakter LoRA   │  🐾 Hayvan/Pet LoRA  │  🎨 Stil Transferi │
│  Yüz eğitimi        │  Evcil hayvan tanıma │  Sanatsal stiller   │
│  Portre datasetleri │  Yaban hayatı       │  Nesne datasetleri   │
└─────────────────────┴──────────────────────┴─────────────────────┘
```

---

## ✨ Temel Özellikler

<div align="center">

### 🤖 **Yapay Zeka Destekli Tespit**

</div>

<table>
<tr>
<td width="33%" align="center">

#### 🎯 YOLOv8
**Hızlı & Doğru**
- En son teknoloji tespit
- Gerçek zamanlı işleme
- GPU hızlandırmalı

</td>
<td width="33%" align="center">

#### 🧠 Ensemble Modu
**3 Yapay Zeka Modeli**
- YOLO + DETR + Faster R-CNN
- Oylama mekanizması
- %95+ doğruluk

</td>
<td width="33%" align="center">

#### ⚡ Turbo Modu
**2-3x Daha Hızlı**
- Toplu işleme
- FP16 çıkarım
- Optimize bellek

</td>
</tr>
</table>

<div align="center">

### 🎨 **Akıllı İşleme**

</div>

| Özellik | Açıklama | Fayda |
|---------|----------|-------|
| **📐 Çoklu Formatlar** | 9:16, 3:4, 1:1, 4:5, 16:9, 4:3 | Her kullanım için mükemmel kırpma |
| **🎯 Akıllı Kırpma** | Baş boşluğu farkındalığı + merkezleme | Profesyonel kalite çerçeveleme |
| **📝 Metin Tespiti** | Otomatik altyazı atlama | Temiz, metinsiz görüntüler |
| **💎 Kalite Puanlama** | Otomatik kalite değerlendirme | Sadece en iyi kareleri kaydet |
| **🗂️ Oto-Organizasyon** | Kişi/hayvan/nesne kategorileri | Eğitime hazır yapı |

<div align="center">

### 🚀 **YENİ: Toplu İşleme**

</div>

```
┌──────────────────────────────────────────────────────────────┐
│  📹 Video 1  →  ✅ İşlendi  →  💾 150 kare kaydedildi       │
│  📹 Video 2  →  ✅ İşlendi  →  💾 200 kare kaydedildi       │
│  📹 Video 3  →  ✅ İşlendi  →  💾 180 kare kaydedildi       │
│                                                              │
│  ✅ TOPLAM: 3 video, 530 kare 5 dakikada!                   │
└──────────────────────────────────────────────────────────────┘
```

**🎉 Sınırsız videoyu tek komutta işleyin!**

---

## 🚀 Hızlı Başlangıç

<details open>
<summary><b>📦 Seçenek 1: Otomatik Kurulum (Windows - Önerilen)</b></summary>

```bash
# Bu dosyalara çift tıklayın:
install.bat          # Her şeyi otomatik kur
run.bat             # GUI modunu başlat
run_batch.bat       # Toplu işlem sihirbazını başlat
```

</details>

<details>
<summary><b>🔧 Seçenek 2: Manuel Kurulum</b></summary>

```bash
# 1. Sanal ortam oluştur
python -m venv venv

# 2. Ortamı aktif et
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Çalıştır!
python main.py
```

</details>

<details>
<summary><b>🐍 Seçenek 3: Direkt Python</b></summary>

```bash
pip install -r requirements.txt
python main.py
```

</details>

---

## 💡 Kullanım Örnekleri

### 🖥️ **GUI Modu** (Başlangıç Seviyesi)

<table>
<tr>
<td width="50%">

#### Tek Video
1. Başlat: `python main.py`
2. Videoyu sürükle-bırak
3. Ayarları düzenle
4. **İşlemi Başlat**'a tıkla

</td>
<td width="50%">

#### Toplu İşleme (YENİ!)
1. Başlat: `python main.py`
2. **Birden fazla video** sürükle-bırak
3. Veya Gözat'ı kullan (Ctrl+Tıkla)
4. **İşlemi Başlat**'a tıkla

</td>
</tr>
</table>

### ⌨️ **CLI Modu** (İleri Seviye)

<details open>
<summary><b>🎯 Tek Video İşleme</b></summary>

```bash
# Basit kullanım
python cli.py video.mp4

# Özel ayarlar
python cli.py video.mp4 -f 1:1 -i 30 -c 0.6

# Yüksek kalite modu
python cli.py video.mp4 -f 1:1 -i 15 -c 0.7 --ensemble --turbo
```

</details>

<details open>
<summary><b>📹 Toplu İşleme (YENİ!)</b></summary>

```bash
# Yöntem 1: Videoları listele
python cli.py video1.mp4 video2.mp4 video3.mp4

# Yöntem 2: Wildcard kullan
python cli.py *.mp4
python cli.py videolar/*.mp4

# Yöntem 3: Toplu işlem sihirbazı
run_batch.bat

# Yöntem 4: Yüksek kalite toplu işlem
python cli.py videolar/*.mp4 -f 1:1 -i 20 --ensemble --turbo
```

</details>

<details>
<summary><b>🎨 Gerçek Dünya Örnekleri</b></summary>

#### Örnek 1: Karakter LoRA Eğitim Dataseti
```bash
python cli.py karakter_video.mp4 -f 1:1 -i 15 -c 0.7 --ensemble --turbo
```
**Sonuç**: 200-500 yüksek kaliteli yüz kırpımı

#### Örnek 2: Evcil Hayvan/Hayvan LoRA Dataseti
```bash
python cli.py evcil_hayvan_videolari/*.mp4 -f 1:1 -i 20 -c 0.6 --ensemble
```
**Sonuç**: Çeşitli açılardan tutarlı hayvan fotoğrafları

#### Örnek 3: Dikey İçerik (TikTok/Reels)
```bash
python cli.py icerik.mp4 -f 9:16 -i 30 --turbo
```
**Sonuç**: Sosyal medya için hazır dikey format kırpımları

#### Örnek 4: Maksimum Kalite (Katı Mod)
```bash
python cli.py video.mp4 -f 1:1 -i 10 -c 0.8 --ensemble --voting-threshold 3
```
**Sonuç**: Sadece 3 yapay zeka modelinin de hemfikir olduğu kareler

#### Örnek 5: Hızlı Önizleme
```bash
python cli.py test.mp4 -f 1:1 -i 90
```
**Sonuç**: 30 saniyede hızlı test

</details>

---

## 🎛️ Parametreler & Ayarlar

<div align="center">

### 📋 **Komple Parametre Referansı**

</div>

| Parametre | Kısa | Seçenekler | Varsayılan | Açıklama |
|-----------|------|------------|------------|----------|
| `videos` | - | dosya yolları | *zorunlu* | 🎬 Tek veya çoklu video dosyaları |
| `--output` | `-o` | yol | `output` | 📁 Çıktı dizini |
| `--format` | `-f` | 9:16, 3:4, 1:1, 4:5, 16:9, 4:3 | `9:16` | 📐 En-boy oranı |
| `--interval` | `-i` | 1-200 | `30` | ⏱️ Her N karede bir işle |
| `--confidence` | `-c` | 0.1-0.95 | `0.5` | 🎯 Tespit eşiği |
| `--padding` | `-p` | 100-1000 | `500` | 📏 Min dolgu (piksel) |
| `--model` | `-m` | n/s/m/l | `yolov8n.pt` | 🤖 YOLO model boyutu |
| `--ensemble` | - | bayrak | KAPALI | 🧠 3-model ensemble aktif et |
| `--ensemble-models` | - | yolo, detr, fasterrcnn | 3'ü de | 🎯 Ensemble için modeller |
| `--voting-threshold` | - | 1-3 | `2` | 🗳️ Min model anlaşması |
| `--turbo` | - | bayrak | **AÇIK** | ⚡ Toplu kare işleme |
| `--no-turbo` | - | bayrak | KAPALI | 🐌 Turbo modunu kapat |
| `--batch-size` | - | 1-16 | `4` | 📦 Toplu başına kare |
| `--no-skip-text` | - | bayrak | KAPALI | 📝 Metin karelerini işle |

<div align="center">

### 🎨 **Önerilen Ön Ayarlar**

</div>

<table>
<tr>
<th>Kullanım Durumu</th>
<th>Komut</th>
<th>Hız</th>
<th>Kalite</th>
</tr>
<tr>
<td>🏆 <b>LoRA Eğitimi (En İyi)</b></td>
<td><code>-f 1:1 -i 15 -c 0.7 --ensemble --turbo</code></td>
<td>⚡⚡</td>
<td>⭐⭐⭐⭐⭐</td>
</tr>
<tr>
<td>⚡ <b>Hızlı Toplama</b></td>
<td><code>-f 1:1 -i 50 --turbo</code></td>
<td>⚡⚡⚡⚡</td>
<td>⭐⭐⭐</td>
</tr>
<tr>
<td>📱 <b>Dikey İçerik</b></td>
<td><code>-f 9:16 -i 30 --turbo</code></td>
<td>⚡⚡⚡</td>
<td>⭐⭐⭐⭐</td>
</tr>
<tr>
<td>💎 <b>Maksimum Kalite</b></td>
<td><code>-f 1:1 -i 10 --ensemble --voting-threshold 3</code></td>
<td>⚡</td>
<td>⭐⭐⭐⭐⭐</td>
</tr>
<tr>
<td>🧪 <b>Hızlı Test</b></td>
<td><code>-f 1:1 -i 90</code></td>
<td>⚡⚡⚡⚡⚡</td>
<td>⭐⭐</td>
</tr>
</table>

---

## 🏗️ Proje Mimarisi

<div align="center">

### 📁 **Dosya Yapısı**

</div>

```
🌾 LoRA-Harvester/
│
├── 🚀 Giriş Noktaları
│   ├── main.py                      # GUI başlatıcı
│   ├── cli.py                       # CLI arayüzü (toplu destek)
│   ├── run.bat                      # Hızlı başlatma scripti
│   └── run_batch.bat                # Toplu işlem sihirbazı
│
├── 🧠 Çekirdek Motor (src/core/)
│   ├── unified_processor.py         # ⭐ Hepsi bir arada işlemci
│   ├── detector.py                  # YOLOv8 tespiti
│   ├── ensemble_detector.py         # Çoklu-model ensemble
│   ├── text_detector.py             # Altyazı tespiti
│   └── cropper.py                   # Akıllı kırpma
│
├── 🎨 Kullanıcı Arayüzü (src/ui/)
│   ├── main_window.py               # PyQt5 GUI (toplu destek)
│   └── translations.py              # TR/EN çeviriler
│
├── 📚 Dokümantasyon
│   ├── README.md                    # Bu dosya
│   ├── CHANGELOG.md                 # Yenilikler
│   ├── QUICKSTART_NEW.md           # Hızlı referans
│   ├── ENSEMBLE.md                  # Ensemble kılavuzu
│   └── OPTIMIZATION.md              # Performans ipuçları
│
└── ⚙️ Yapılandırma
    ├── config.yaml                  # Ayarlar
    ├── requirements.txt             # Bağımlılıklar
    └── yolov8n.pt                  # Yapay zeka modeli
```

<div align="center">

### 🔄 **İşleme Hattı**

</div>

```
┌─────────────┐
│ 📹 Video(lar│
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ 🤖 Yapay Zeka Tespiti│──┐
│ • YOLOv8            │  │ Ensemble Modu
│ • DETR (opsiyonel)  │◄─┤ (3 model oylar)
│ • Faster R-CNN      │  │
└──────┬──────────────┘──┘
       │
       ▼
┌─────────────────────┐
│ 📝 Metin Tespiti    │
│ Altyazıları atla?   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ ✂️ Akıllı Kırpma    │
│ • Baş boşluğu hesap │
│ • Özne merkezleme   │
│ • Format ayarlama   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 💎 Kalite Kontrolü  │
│ Puan: 0.0 - 1.0     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 💾 Kaydet & Organize │
│ • persons/          │
│ • animals/          │
│ • objects/          │
└─────────────────────┘
```

---

## ⚙️ Yapılandırma

<details>
<summary><b>📝 config.yaml Ayarları</b></summary>

```yaml
# Tespit Ayarları
detection:
  model_size: "yolov8n.pt"        # n=hızlı, s=dengeli, m/l=doğru
  confidence: 0.5                  # 0.1-0.95
  
# Kırpma Ayarları
cropping:
  default_format: "1:1"            # LoRA eğitimi için en iyi
  min_padding: 500                 # Özne etrafında piksel
  
# Metin Tespiti
text_detection:
  enabled: true                    # Altyazı karelerini atla
  languages: ["en", "tr"]          # Desteklenen diller
  
# Ensemble Modu
ensemble:
  enabled: false                   # CLI'da --ensemble ile aktif et
  voting_threshold: 2              # Min anlaşma (1-3)
  models: ["yolo", "detr", "fasterrcnn"]
  
# Performans
performance:
  turbo_mode: true                 # Toplu işleme
  batch_size: 4                    # Toplu başına kare
  use_fp16: true                   # Yarı hassasiyet (destekleniyorsa)
```

</details>

---

## 🔧 Sistem Gereksinimleri

<div align="center">

### 💻 **Donanım Gereksinimleri**

</div>

<table>
<tr>
<th></th>
<th>Minimum</th>
<th>Önerilen</th>
</tr>
<tr>
<td><b>CPU</b></td>
<td>Intel i5 / AMD Ryzen 5</td>
<td>Intel i7 / AMD Ryzen 7</td>
</tr>
<tr>
<td><b>RAM</b></td>
<td>8 GB</td>
<td>16 GB</td>
</tr>
<tr>
<td><b>GPU</b></td>
<td>Opsiyonel (CPU modu)</td>
<td>NVIDIA GTX 1060 6GB</td>
</tr>
<tr>
<td><b>Depolama</b></td>
<td>10 GB boş</td>
<td>20 GB boş</td>
</tr>
</table>

<div align="center">

### 📦 **Yazılım Gereksinimleri**

</div>

| Yazılım | Versiyon | Gerekli | Notlar |
|---------|----------|---------|--------|
| **Python** | 3.8 - 3.11 | ✅ Evet | Python 3.12 henüz desteklenmiyor |
| **CUDA Toolkit** | 11.8+ | ⚠️ Sadece GPU | NVIDIA GPU hızlandırma için |
| **Tesseract OCR** | En son | ⚠️ Opsiyonel | Gelişmiş metin tespiti için |
| **Windows** | 10/11 | ✅ Önerilen | Linux/Mac da destekleniyor |

---

## 🚀 Performans Kılavuzu

<div align="center">

### ⚡ **Hız Karşılaştırması**

</div>

| Mod | GPU | CPU | 10dk Video |
|-----|-----|-----|------------|
| **Standart (YOLO)** | ~30 FPS | ~5 FPS | 20-30 sn |
| **Turbo (YOLO)** | ~60 FPS | ~10 FPS | 10-15 sn |
| **Ensemble (3 model)** | ~10 FPS | ~2 FPS | 60-90 sn |
| **Ensemble + Turbo** | ~20 FPS | ~4 FPS | 30-45 sn |

<div align="center">

### 💡 **Optimizasyon İpuçları**

</div>

<table>
<tr>
<td width="33%">

#### 🐌 Çok Yavaş mı?
- ✅ Turbo modunu aktif et
- ✅ Kare aralığını artır
- ✅ Küçük YOLO modeli kullan
- ✅ Ensemble modunu kapat

</td>
<td width="33%">

#### 💾 Bellek Doldu mu?
- ✅ Toplu boyutunu azalt
- ✅ Kare aralığını artır
- ✅ CPU modunu kullan
- ✅ Diğer programları kapat

</td>
<td width="33%">

#### 📉 Kalite Düşük mü?
- ✅ Ensemble modunu aktif et
- ✅ Kare aralığını azalt
- ✅ Confidence'ı artır
- ✅ Büyük YOLO modeli kullan

</td>
</tr>
</table>

---

## 🎬 Kullanım Alanları & Uygulamalar

<div align="center">

### 🎯 **Gerçek Dünya Uygulamaları**

</div>

<table>
<tr>
<td align="center" width="25%">

### 👤 Karakter LoRA
**Stable Diffusion**

Özel karakter modelleri eğit

✅ Yüz tutarlılığı  
✅ Çoklu açılar  
✅ Çeşitli ifadeler  

</td>
<td align="center" width="25%">

### 🐾 Hayvan/Pet LoRA
**Evcil Hayvan Tanıma**

Pet-özel modeller oluştur

✅ Pet portreleri  
✅ Cins eğitimi  
✅ Yaban hayatı datasetleri  

</td>
<td align="center" width="25%">

### 🎨 Stil Transferi
**Sanatsal Yapay Zeka**

Stil modelleri eğit

✅ Sanat stilleri  
✅ Nesne tutarlılığı  
✅ Sahne datasetleri  

</td>
<td align="center" width="25%">


</td>
</tr>
</table>

---

## 🐛 Sorun Giderme

<details>
<summary><b>❌ CUDA Kullanılamıyor</b></summary>

**Problem**: CPU'da çalışıyor, yavaş performans

**Çözüm**:
```bash
# CUDA-etkin PyTorch kur
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU'yu doğrula
python -c "import torch; print(torch.cuda.is_available())"
```

</details>

<details>
<summary><b>💾 Bellek Yetersiz Hatası</b></summary>

**Problem**: GPU/RAM belleği doldu

**Çözümler**:
```bash
# Seçenek 1: Toplu boyutunu azalt
python cli.py video.mp4 --batch-size 2

# Seçenek 2: Kare aralığını artır
python cli.py video.mp4 -i 60

# Seçenek 3: Küçük model kullan
python cli.py video.mp4 -m yolov8n.pt

# Seçenek 4: Ensemble'ı kapat
python cli.py video.mp4  # --ensemble bayrağı yok
```

</details>

<details>
<summary><b>📉 Hiç Kare Kaydedilmedi / Çok Az Kare</b></summary>

**Problem**: Çıktı klasörü boş veya çok az kare

**Çözümler**:
```bash
# Seçenek 1: Confidence eşiğini düşür
python cli.py video.mp4 -c 0.3

# Seçenek 2: Daha fazla kare işle
python cli.py video.mp4 -i 15

# Seçenek 3: Metin karelerini dahil et
python cli.py video.mp4 --no-skip-text

# Seçenek 4: Video kalitesini kontrol et
# Videonun net ve iyi aydınlatılmış olduğundan emin ol
```

</details>

<details>
<summary><b>🤖 Ensemble Modu Çok Yavaş</b></summary>

**Problem**: Ensemble çok uzun sürüyor

**Çözümler**:
```bash
# Seçenek 1: Sadece 2 model kullan
python cli.py video.mp4 --ensemble --ensemble-models yolo detr

# Seçenek 2: Kare aralığını artır
python cli.py video.mp4 --ensemble -i 60

# Seçenek 3: Turbo'yu aktif et (zaten değilse)
python cli.py video.mp4 --ensemble --turbo

# Seçenek 4: GPU kullan
# Ensemble modu gerçekten GPU gerektirir
```

</details>

<details>
<summary><b>📝 Çok Fazla Yanlış Tespit</b></summary>

**Problem**: Kötü/yanlış tespitler kaydediliyor

**Çözümler**:
```bash
# Seçenek 1: Katı oylamayla ensemble kullan
python cli.py video.mp4 --ensemble --voting-threshold 3

# Seçenek 2: Confidence'ı artır
python cli.py video.mp4 -c 0.7

# Seçenek 3: İkisini birleştir
python cli.py video.mp4 --ensemble --voting-threshold 3 -c 0.7
```

</details>

---

## 📚 Ek Kaynaklar

<div align="center">

### 📖 **Dokümantasyon**

</div>

| Belge | Açıklama | Link |
|-------|----------|------|
| 🚀 **QUICKSTART_NEW.md** | Hızlı referans & örnekler | [Görüntüle](QUICKSTART_NEW.md) |
| 📝 **CHANGELOG.md** | Son sürümdeki yenilikler | [Görüntüle](CHANGELOG.md) |
| 🤖 **ENSEMBLE.md** | Ensemble modu detaylı kılavuz | [Görüntüle](ENSEMBLE.md) |
| ⚡ **OPTIMIZATION.md** | Performans optimizasyon ipuçları | [Görüntüle](OPTIMIZATION.md) |

<div align="center">

### 🔗 **Harici Linkler**

</div>

- 🐙 **GitHub Deposu**: [AllastorV/LoRA-Harvester](https://github.com/AllastorV/LoRA-Harvester)
- 📧 **Sorunlar & Destek**: [GitHub Issues](https://github.com/AllastorV/LoRA-Harvester/issues)
- 🌟 **GitHub'da Yıldızla**: [Bize ⭐ ver](https://github.com/AllastorV/LoRA-Harvester)

---

## 📝 Lisans & Katkılar

<div align="center">

### 📜 **Lisans**

**GNU Genel Kamu Lisansı v3.0**

Copyleft gereklilikleriyle kişisel ve ticari kullanım için ücretsiz

</div>

```
Bu program özgür bir yazılımdır: Free Software Foundation tarafından
yayınlanan GNU Genel Kamu Lisansı'nın 3. veya daha sonraki bir sürümünün
şartları altında yeniden dağıtabilir ve/veya değiştirebilirsiniz.
```

<div align="center">

### 🙏 **Teşekkürler**

</div>

<table>
<tr>
<td align="center">
<b>🤖 YOLOv8</b><br>
Ultralytics tarafından<br>
Nesne Tespiti
</td>
<td align="center">
<b>🧠 DETR</b><br>
Facebook/Meta tarafından<br>
Transformer Tespiti
</td>
<td align="center">
<b>🔍 Faster R-CNN</b><br>
Torchvision tarafından<br>
Bölge-tabanlı Tespit
</td>
<td align="center">
<b>📝 EasyOCR</b><br>
JaidedAI tarafından<br>
Metin Tespiti
</td>
</tr>
<tr>
<td align="center">
<b>🔥 PyTorch</b><br>
Derin Öğrenme<br>
Framework'ü
</td>
<td align="center">
<b>🎨 OpenCV</b><br>
Video İşleme<br>
Kütüphanesi
</td>
<td align="center">
<b>🖼️ PyQt5</b><br>
GUI Framework'ü<br>
Arayüz
</td>
<td align="center">
<b>💜 Topluluk</b><br>
Katkıda Bulunanlar<br>
& Kullanıcılar
</td>
</tr>
</table>

---

## 🤝 Katkıda Bulunma

<div align="center">

**🎉 Katkılarınızı bekliyoruz!**

</div>

1. 🍴 Depoyu fork'layın
2. 🌿 Feature branch'inizi oluşturun (`git checkout -b feature/HarikaOzellik`)
3. 💾 Değişikliklerinizi commit'leyin (`git commit -m 'Harika özellik ekle'`)
4. 📤 Branch'e push yapın (`git push origin feature/HarikaOzellik`)
5. 🎁 Pull Request açın

---

## 📧 Destek

<div align="center">

### 💬 **Yardıma mı İhtiyacınız Var?**

</div>

<table>
<tr>
<td align="center" width="33%">

### 📖 Dokümantasyon
Önce kılavuzlara bakın

[Dokümanları Oku](#documentation)

</td>
<td align="center" width="33%">

### 🐛 Hata Bildirimi
Bir sorun mu buldunuz?

[GitHub'da Bildir](https://github.com/AllastorV/LoRA-Harvester/issues)

</td>
<td align="center" width="33%">

### 💡 Özellik İsteği
Bir fikriniz mi var?

[GitHub'da Önerin](https://github.com/AllastorV/LoRA-Harvester/issues)

</td>
</tr>
</table>

---

<div align="center">

## 🌟 Faydalı Bulduysanız Yıldız Verin! / Star if You Find it Useful!

[![GitHub stars](https://img.shields.io/github/stars/AllastorV/LoRA-Harvester?style=social)](https://github.com/AllastorV/LoRA-Harvester)

**Made with 💜 by [AllastorV](https://github.com/AllastorV)**

```
┌─────────────────────────────────────────────────────────────┐
│  🌾 LoRA-Harvester - AI-Powered Dataset Collection         │
│  🚀 Fast • 🎯 Accurate • 📦 Batch Processing • 🤖 Ensemble │
│  ⭐ Star on GitHub • 🐛 Report Issues • 💡 Contribute      │
└─────────────────────────────────────────────────────────────┘
```

**[⬆ Back to top](#-lora-harvester)**


---

<div align="center">

**[GitHub @AllastorV](https://github.com/AllastorV)**

*Faydalı bulursanız ⭐ yıldız verin! / Star ⭐ if you find it useful!*

</div>









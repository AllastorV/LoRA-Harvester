# 🌾 LoRA-Harvester - Performance Optimization Guide / Performans Optimizasyon Kılavuzu

[English](#english) | [Türkçe](#turkce)

---

<a name="english"></a>
## 🇬🇧 Performance Guide

### 🚀 Turbo Mode

**2-3x faster processing** with batch processing and GPU optimizations.

```bash
# Enable turbo mode
python cli.py video.mp4 --turbo

# Custom batch size
python cli.py video.mp4 --turbo --batch-size 8

# Turbo + Ensemble
python cli.py video.mp4 --turbo --ensemble
```

### 🎯 Optimization Strategies

| Strategy | Speed Gain | Quality Impact |
|----------|------------|----------------|
| **Increase interval** `-i 60` | 2x | Low |
| **Enable turbo** `--turbo` | 2-3x | None |
| **Skip text check** `--no-skip-text` | 1.5x | Low |
| **Use smaller model** `-m yolov8n.pt` | 1.5x | Medium |
| **Reduce padding** `-p 200` | 1.2x | Low |

### ⚙️ Recommended Settings

**Fast Processing (15-20 FPS)**
```bash
python cli.py video.mp4 -i 90 --turbo --batch-size 8 --no-skip-text
```

**Balanced (8-12 FPS)**
```bash
python cli.py video.mp4 -i 60 --turbo --batch-size 4
```

**High Quality (4-6 FPS)**
```bash
python cli.py video.mp4 -i 30 --turbo --ensemble --batch-size 2
```

### 🔧 GPU Settings

```python
# Check GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

**GPU Memory Management:**
- **4GB**: batch-size 2, single mode
- **6GB**: batch-size 4, ensemble mode
- **8GB+**: batch-size 8, ensemble + turbo

### 💡 Performance Tips

1. **Use SSD**: 20% faster file I/O
2. **Close other apps**: More GPU memory
3. **Update drivers**: NVIDIA latest drivers
4. **Use H.264 codec**: Faster decoding
5. **Preprocess video**: Convert to MP4 if needed

---

<a name="turkce"></a>
## 🇹🇷 Performans Kılavuzu

### 🚀 Turbo Modu

**2-3x daha hızlı işleme** toplu işleme ve GPU optimizasyonları ile.

```bash
# Turbo modunu aktif et
python cli.py video.mp4 --turbo

# Özel toplu boyutu
python cli.py video.mp4 --turbo --batch-size 8

# Turbo + Topluluk
python cli.py video.mp4 --turbo --ensemble
```

### 🎯 Optimizasyon Stratejileri

| Strateji | Hız Kazancı | Kalite Etkisi |
|----------|-------------|---------------|
| **Aralığı artır** `-i 60` | 2x | Düşük |
| **Turbo aç** `--turbo` | 2-3x | Yok |
| **Metin kontrolünü atla** `--no-skip-text` | 1.5x | Düşük |
| **Küçük model** `-m yolov8n.pt` | 1.5x | Orta |
| **Dolguyu azalt** `-p 200` | 1.2x | Düşük |

### ⚙️ Önerilen Ayarlar

**Hızlı İşleme (15-20 FPS)**
```bash
python cli.py video.mp4 -i 90 --turbo --batch-size 8 --no-skip-text
```

**Dengeli (8-12 FPS)**
```bash
python cli.py video.mp4 -i 60 --turbo --batch-size 4
```

**Yüksek Kalite (4-6 FPS)**
```bash
python cli.py video.mp4 -i 30 --turbo --ensemble --batch-size 2
```

### 🔧 GPU Ayarları

```python
# GPU kontrolü
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

**GPU Bellek Yönetimi:**
- **4GB**: batch-size 2, tekli mod
- **6GB**: batch-size 4, topluluk modu
- **8GB+**: batch-size 8, topluluk + turbo

### 💡 Performans İpuçları

1. **SSD kullan**: %20 daha hızlı dosya I/O
2. **Diğer uygulamaları kapat**: Daha fazla GPU belleği
3. **Sürücüleri güncelle**: NVIDIA en son sürücüler
4. **H.264 codec kullan**: Daha hızlı decode
5. **Videoyu ön-işle**: Gerekirse MP4'e çevir

### 🎯 Hız Karşılaştırması

| Mod | Ayarlar | FPS | Kalite |
|-----|---------|-----|--------|
| **Standart** | Varsayılan | 3-5 | 100% |
| **Hızlı** | `-i 60 --turbo` | 10-15 | 95% |
| **Çok Hızlı** | `-i 90 --turbo --no-skip-text` | 15-20 | 90% |
| **Hassas** | `-i 30 --ensemble` | 4-6 | 105% |

### 🐛 Sorun Giderme

**Yavaş İşleme:**
- GPU kullanıldığından emin olun
- Kare aralığını artırın (`-i 90`)
- Turbo modunu açın (`--turbo`)
- Diğer GPU kullanan uygulamaları kapatın

**Bellek Hatası:**
- Toplu boyutu azaltın (`--batch-size 2`)
- Daha küçük model kullanın (`yolov8n.pt`)
- Topluluk modunu kapatın
- Video çözünürlüğünü düşürün

**Donma:**
- İlk çalıştırmada modeller yükleniyor
- Lazy loading aktif
- 2-3 saniye bekleyin

---

## 📊 Benchmark Results

### Test System
- GPU: NVIDIA RTX 3060 (12GB)
- CPU: Intel i7-10700K
- RAM: 16GB
- Video: 1080p, 30fps, H.264

### Results

| Configuration | FPS | GPU Usage | VRAM |
|---------------|-----|-----------|------|
| Single + Normal | 5 | 60% | 2GB |
| Single + Turbo | 12 | 80% | 3GB |
| Ensemble + Normal | 3 | 70% | 4GB |
| Ensemble + Turbo | 8 | 90% | 5GB |

---

## 🚀 Best Practices

### For Content Creators
```bash
# Balanced speed and quality
python cli.py video.mp4 -f 9:16 -i 45 --turbo --batch-size 6
```

### For Quick Tests
```bash
# Maximum speed
python cli.py video.mp4 -i 120 --turbo --batch-size 8 --no-skip-text
```

### For Production
```bash
# Maximum quality
python cli.py video.mp4 -i 20 -c 0.7 --ensemble --voting-threshold 2
```

---

<div align="center">

**⚡ Optimized for Speed and Quality**

**Hız ve Kalite için Optimize Edildi**

</div>

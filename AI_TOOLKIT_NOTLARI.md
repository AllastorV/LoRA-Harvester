# LoRA-Harvester v5 · İki eğitim motoru

**Teslim:** `2026.09.22-v5-toolkit1`\
**Temel:** Bu konuşmadaki son `LoRA-Harvester-AI-Kontrol-Tam.zip` (`2026.09.22-ai1`).

> **Training → Eğitim motoru → Kohya / AI Toolkit**\
> Kohya kaldırılmadı. Ostris AI Toolkit ayrı bir eğitim motoru olarak aynı masaüstü programına eklendi. Bu paket kaynak koddur; model ağırlığı, AI Toolkit deposu, Python ortamı veya EXE içermez.

## 1. Güncelleme

**Çalışan kurulum:** Programı ve eğitim işlerini kapat. Proje klasörünü yedekle. `LoRA-Harvester-V5-AI-Toolkit-Guncelleme.zip` içeriğini `main.py` dosyasının bulunduğu klasöre çıkar; klasörleri birleştir, kaynak dosyalarını değiştir. Eski `src` klasörünü tamamen silme. `run.bat` ile aç.

Bu kaynak güncellemesi Harvester'ın Python, PyTorch, CUDA veya diğer modellerini yeniden kurmaz. Güncelleme ZIP'inde `config`, `data`, `models`, `output`, `venv`, `.venv`, `logs` ve `kohya_ss` yoktur. Kıyafet kütüphanen ve kişisel ayarların korunur.

**Yeni kurulum:** Tam ZIP'i yeni klasöre çıkar. `install.bat` Harvester kurulumudur; `run.bat` programı açar. Tam paketin varsayılan `config` dosyalarını çalışan kişisel ayarların üzerine kopyalama. `run_classic.bat` korunur.

**AI Toolkit ayrı kurulumdur.** Harvester kurulum sihirbazı bu güncellemede AI Toolkit'i kurmaz; çalışan Harvester ortamına Toolkit paketlerini pip ile ekleme.

## 2. AI Toolkit'i bağlama

1. Resmî [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit#readme) kurulumunu ayrı klasör ve ayrı Python ortamıyla yap. Kurulum seçenekleri upstream projeye aittir; Harvester gizli kurulum, otomatik güncelleme veya yönetici yetkisi istemez.
2. Harvester'da **Training → Eğitim motoru → AI Toolkit · Ostris** seç.
3. **Toolkit bağlantısı** sekmesinde `run.py` içeren kök klasörü seç. Örnek: `D:\AI\ai-toolkit`.
4. O kurulumun Python çalıştırıcısını seç. Örnek: `D:\AI\ai-toolkit\venv\Scripts\python.exe`. `venv` / `.venv` bilinen yerlerdeyse önerilir; farklı yerde kurulu ortam elle seçilebilir. Harvester'ın kendi Python ortamı ve global Python reddedilir.
5. **Toolkit ortamını kontrol et**: seçili Python, temel modüllerin varlığı ve CUDA aygıtları kontrol edilir. Bu kontrol model indirmez veya eğitim yapmaz; bütün bağımlılık kombinasyonlarını doğrulanmış saymaz.

Windows'ta kurulu Toolkit'in NVIDIA/CUDA desteği ayrıca çalışmalıdır. Harvester'da GPU çalışması, başka bir venv'de kurulu Toolkit'in de GPU kullanabildiği anlamına gelmez. Bir NVIDIA aygıtı seçilir; çok-GPU/distributed eğitim bu arayüzün kapsamında değildir.

## 3. Datasetten eğitime

**Temel** sekmesinde dataset, çıktı ana klasörü, eğitim adı, model ailesi ve ana modeli seç. Ortak Studio dataset'i açıksa boş Toolkit dataset alanı bu klasörden doldurulur; önceden seçilmiş başka bir klasör zorla değiştirilmez.

| Model ailesi | Bu arayüzdeki profil |
|---|---|
| SDXL / Illustrious | SDXL tabanlı görsel LoRA. Varsayılan 1024; yerel `.safetensors`, Diffusers klasörü veya Hugging Face `owner/repository` kimliği. Modelin gerçekten uyumlu olması gerekir. |
| Stable Diffusion 1.5 | SD1.5 tabanlı görsel LoRA. Aile seçildiğinde başlangıç çözünürlüğü 512. |
| FLUX.1-dev | Diffusers klasörü veya uygun HF kimliği; BF16, isteğe bağlı quantization; text encoder kapalı, min-SNR/noise offset sıfır. |

Bu liste **upstream AI Toolkit'in bütün model ailesi desteği değildir**. Video modelleri, yeni mimariler, LoKr/LoCon, çok-GPU ve tüm upstream gelişmiş seçenekler eklenmedi. Model ailesi dosya adına bakılarak otomatik tahmin edilmez. FLUX profilinin mevcut GPU belleğine sığdığı burada doğrulanmadı.

**YAML oluştur** verileri işçi thread'inde tarar ve yerel bir iş paketi oluşturur. Ardından **YAML önizlemesi** görünür. Kontrol ettikten sonra **Eğitimi başlat** ile açık onay ver. Başlatmadan önce kaynak görsel/caption ve yapılandırma tekrar doğrulanır; değişmiş bir dataset için eski önizleme kullanılmaz.

Kohya için TOML/epoch, AI Toolkit için native YAML/step kullanılır. Bir motorun ayarları diğerine körlemesine çevrilmez; Kohya formları korunur.

## 4. Kullanılabilir ayarlar

**Temel:** toplam optimizer step, çözünürlük, batch, LR, LoRA rank/alpha, checkpoint aralığı ve saklanacak ara checkpoint sayısı.

**Gelişmiş:** gradient accumulation, AdamW8bit/AdamW/Adafactor, FP16/BF16/FP32, gradient checkpointing, latent disk cache, seed, CUDA aygıtı, data-loader işçisi, alt klasör/repeat davranışı, caption zorunluluğu, shuffle, keep tokens, caption dropout, yatay flip, text encoder aç/kapat, min-SNR, noise offset ve network dropout. Uyumsuz FLUX kontrolleri kapatılır.

- LR alanı **10 ondalık basamak** destekler; `0.000005` gibi değerler sıfıra yuvarlanmaz.
- Upstream'in incelenen LoRA yolunda text encoder optimizer parametreleri de ortak `train.lr` kullanıyor. Bu nedenle **ayrı text encoder LR alanı eklenmedi**; uygulanmayan bir değer varmış gibi gösterilmiyor. Text encoder varsayılan kapalıdır. [3]
- `gradient_accumulation` optimizer-step semantiğiyle yazılır; eski `gradient_accumulation_steps` 1 tutulur. Epoch sayısı gibi sunulmaz. [2]
- Başlangıç ayarları bir eğitim reçetesi/kalite garantisi değildir. Örneğin 3000 step bütün datasetler için ideal ilan edilmez.
- Checkpoint saklama sınırı, upstream tarafından eski ara kayıtların silinmesine yol açabilir. İhtiyaç duyduğun sayıda kayıt tutacak değeri seç; eğitim başlatma onayında bu uyarı gösterilir.

**Örnek üretimi:** her satıra bir prompt, örnek aralığı, diffusion step ve guidance. Liste boşsa sampling kapalıdır. Örnek üretimi ek süre/VRAM tüketir. Seed sabittir; farklı donanım/sürümde bit düzeyinde aynı görüntü garantisi yoktur.

## 5. Caption, master tag ve tekrarların korunması

Dataset yeniden kodlanmaz, görseller kopyalanmaz ve caption dosyaları değiştirilmez. Desteklenen girdiler PNG, JPG/JPEG ve durağan WebP'dir. GIF/TIFF/BMP vb. bulunduğunda sessiz atlamak yerine uygun biçimde ayrı dışa aktarım istenir.

Her görsel için mutlak dosya yolu ve mevcut caption metni, Toolkit'in gerçek JSON dataset biçimiyle indekse yazılır:

```json
{
  "D:\\dataset\\image.png": {
    "caption": "shhooldress, serafaku, blue skirt"
  }
}
```

Bu JSON çıktıdaki özel tag yazımını kendiliğinden düzeltmez. Varsayılan shuffle kapalıdır; açıldığında `keep_tokens=1` ilk master tagi korumak için başlangıç ayarıdır. Eğitim modelinin tokenizer/caption işleme davranışı upstream'e aittir; bu indeks, orijinal `.txt` dosyasını değiştirmez.

`4_concept` gibi Kohya klasörlerinde tekrar sayısı isteğe bağlı olarak korunur. Her görsel yalnız bir gruba girer; üst ve alt klasörler iki kere dataset olarak eklenmez. Klasör adındaki tekrar sadece örneklerin ağırlığını etkiler; toplam step sınırını kendiliğinden artırmaz.

Eksik/boş caption varsayılan hata verir. Bu korumayı bilerek kapatınca boş captionlar açık uyarıyla kabul edilir. Bozuk UTF-8, 1 MiB'ı aşan caption, okunamayan görsel ve aynı caption adını paylaşan iki görsel reddedilir. Hariç/metadata klasörleri taranmaz.

**Önbellek sınırı:** AI Toolkit, latent cache açıkken kaynak görsellerin yanında kendi cache dosyalarını oluşturabilir. “Orijinal görüntü/caption içeriği değiştirilmez” ile “dataset dizinine hiçbir yan dosya yazılmaz” aynı şey değildir. Toolkit harici ve güvenilen kod olarak çalıştırılır; güvenlik sandbox'ı değildir.

## 6. Başlatma, durdurma ve devam

Eğitim ayrı Toolkit Python sürecinde çalışır. Günlük ve step ilerlemesi GUI'ye kuyruklu sinyallerle gelir; GUI ana iş parçacığında model yüklenmez. Aynı uygulamadaki veri yazıcıları/diğer eğitim işleri eğitim sırasında kilitlenir. Çalışırken motor seçimi değiştirilemez. AI kontrolünün veri değiştiren çağrıları da aktif training işini dikkate alır.

**Güvenli durdur:** mevcut eğitim adımı ile onun örnek/checkpoint işlemi tamamlandıktan sonraki upstream `end_step_hook` noktasında kesilme istenir. Kaydetmenin ortasına otomatik SIGTERM gönderilmez. Modelin ilk yüklemesi veya cache hazırlığı sürüyorsa bu noktaya ulaşmak gerekebilir. Son kaydedilmiş checkpoint korunur; o kayıttan sonraki adımların ayrıca kaydedildiği söylenmez. [3][4]

**Zorla durdur…** ayrıca onay ister. Yanıt vermeyen süreci sonlandırır; checkpoint yazımını kesebilir. Zorla durdurulan iş başarılı gösterilmez. Eğitim sırasında Harvester'ı kapatma isteği de normalde işçilerin gerçekten bitmesini bekler.

**Devam / başarısız işi yeniden dene:** yalnız aynı Harvester iş kaydı, aynı dataset ve uyumlu ayar imzası bulunan çıktı klasöründe kullanılabilir. Toplam step artırılabilir, sampling/kayıt sıklığı değiştirilebilir. Sıradan dosyalarla dolu, kayıtsız çıktı üzerine yazılmaz. Toolkit mevcut checkpointini kendi resume mekanizmasıyla seçer; Harvester baştan sona bit-eşdeğer devam garantisi vermez. [1]

Yerel model değişikliği dosya boyutu/zamanı ve küçük JSON yapılandırmalarının hash'leriyle kontrol edilir; çok-GB model ağırlıklarının tamamı her seferinde hash'lenmez. HF repository kimliği sabit bir revision hash'i olarak pinlenmez. Aynı kimlikte uzaktan değişmiş ağırlıklar veya elle değiştirilmiş checkpointler için tam bütünlük/reprodüksiyon garantisi verilmez.

## 7. İndirme ve gizlilik

**Eksik model dosyalarını indirmeye izin ver** ilk açılışta kapalıdır ve yeni oturumda yeniden izin ister. Kapalıyken Hugging Face/Transformers offline değişkenleri ayarlanır; gereken cache yoksa eğitim açık hatayla biter. Açıkken erişim yetkisi/lisansı uygun eksik model dosyaları upstream tarafından indirilebilir.

YAML'de `push_to_hub=false`, `use_wandb=false`; otomatik model paylaşımı eklenmedi. Harvester yeni bir API anahtarı istemez veya anahtarları ayar dosyasına yazmaz. Toolkit'in kendi ortam değişkenleri/yerel erişim kimlikleri kullanılabilir. Offline bayrağı işletim sistemi ağ güvenlik duvarı değildir; güvenilmeyen Toolkit kodunu sınırlayan bir sandbox olarak değerlendirilmemelidir.

## 8. Dosya düzeni

```text
LoRA-Harvester-main/
├─ src/training/ai_toolkit_config.py       # doğrulama, dataset indeksleri, native YAML
├─ src/training/ai_toolkit_runner.py       # ayrı venv, probe, süreç/çıktı kilidi, günlük
├─ src/ui/training_hub_page.py            # Kohya / AI Toolkit seçici
├─ src/ui/ai_toolkit_training_page.py     # gerçek PyQt5 formu ve worker
├─ scripts/run_ai_toolkit.py              # upstream CLI + yalnız süreç içi adım hook'u
└─ data/ai_toolkit_training.json          # kullanıcı makinesinde oluşturulan ayarlar

Seçilen çıktı ana klasörü/
├─ .lh-toolkit-jobs/
│  ├─ egitim_adi.lock
│  └─ egitim_adi-<benzersiz-id>/
│     ├─ training.yaml
│     ├─ dataset-r1.json                  # gerekiyorsa farklı repeat grupları
│     ├─ harvester-job.json
│     └─ run-<benzersiz-id>.log
└─ egitim_adi/
   ├─ .lh-toolkit-run.json
   ├─ *.safetensors
   └─ samples/                           # sampling açıksa, upstream oluşturur
```

YAML/JSON içindeki yollar mutlaktır; iş paketini başka makineye taşıdıktan sonra yeniden oluştur. Dataset/caption ve çıktı dizinlerini eğitim boyunca başka programlarla değiştirme. Harvester'ın çıktı kilidi başka programların aynı klasöre yazmasını engellemez.

Kohya'nın Qt işçisi artık yalnız eğitim başlatıldığı için bitmiş sayılmaz; gerçek subprocess izleme thread'i tamamlanana kadar yaşar. Kapanış ve iki motor arasındaki iş/kaynak kontrolleri bu yaşam döngüsünü kullanır.

## 9. Test kapsamı

**Son regresyon: 728 test geçti, 40 Qt testi atlandı, başarısız test yok.** 626 alt-kontrol ayrı üst düzey test sayısına eklenmez. Yeni kontrollerin 107’si çalıştı, 5 gerçek Qt testi ortamda PyQt5 olmadığı için atlandı.

Önceki sürümün **621 çalışan testine** ek testler yazıldı. Tarihsel test raporları orijinal arşivde korunur; bu pakette tekrar çalıştırılabilir test kaynakları `tests/` altındadır.

Tam arşiv temiz çıkarıldıktan sonra testler tekrar geçti. Güncelleme, temel sürüme uygulandı ve kullanıcı kontrol dosyaları değişmedi. İlk overlay doğrulaması zaman aşımıyla tamamlanamadı; aynı kaynak yeniden çalıştırıldığında testler geçti. Bu aralıklı zaman aşımının nedeni belirlenmedi; tarihsel paket kontrol kaydı orijinal arşivdedir.

Yeni testlerde native alan eşlemesi, model aileleri, gerçek PNG/TXT dosyaları, literal caption, repeat/alt klasörler, bozuk girdiler, eski YAML/dataset reddi, yerel model değişikliği, kaynak koruma, aynı çıktı kilidi, günlük/iptal ve resume kuralları çalıştırıldı. Gerçek geçici bir Python venv'sinden **taklit Toolkit ve taklit torch** alt süreci başlatıldı. Bu, gerçek diffusion eğitimi veya gerçek GPU testi değildir.

**Doğrulanmayanlar:** Windows pencere çizimi/DPI, kurulu AI Toolkit'in bütün bağımlılıkları, gerçek SDXL/Illustrious/FLUX eğitimi, RTX 3070 Ti VRAM/süre/kalite, checkpoint ağırlıklarının matematiksel doğruluğu. PyQt5 mevcut olmadığı için gerçek Qt testleri atlandı; kaynak-metot testleri pencere testi yerine sayılmadı. Mevcut MCP bağlantısı korunmuştur; bu güncelleme ayrıca bir MCP “eğitim başlat” aracı eklemez.

```powershell
venv\Scripts\python.exe run_tests.py -q -ra
```

## 10. Teknik dayanaklar

22 Eylül 2026'da erişilen upstream kaynaklara göre uygulanmıştır. Kaynaklar değişirse bu arayüzün açık kapsamı da yeniden doğrulanmalıdır; her gelecek Toolkit sürümüyle uyumluluk iddiası değildir.

1. Ostris AI Toolkit, kurulum ve CLI/resume: https://github.com/ostris/ai-toolkit
2. Gerçek yapılandırma alanları: https://github.com/ostris/ai-toolkit/blob/main/toolkit/config_modules.py
3. LoRA optimizer parametreleri, kayıt ve adım sonu: https://github.com/ostris/ai-toolkit/blob/main/jobs/process/BaseSDTrainProcess.py
4. CLI hata/durdurma davranışı: https://github.com/ostris/ai-toolkit/blob/main/run.py
5. JSON dataset yükleme: https://github.com/ostris/ai-toolkit/blob/main/toolkit/data_loader.py
6. JSON caption nesnesi: https://github.com/ostris/ai-toolkit/blob/main/toolkit/dataloader_mixins.py
7. Resmî FLUX örneği: https://github.com/ostris/ai-toolkit/blob/main/config/examples/train_lora_flux_24gb.yaml
8. CUDA aygıt seçimi: https://github.com/huggingface/accelerate/blob/main/src/accelerate/state.py

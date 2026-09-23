# LoRA-Harvester — Studio geliştirme paketi

**Teslim:** `2026.09.22-r2` · **Tarih:** 22 Eylül 2026\
**Temel:** Bu konuşmada teslim edilen `LoRA-Harvester-Final.zip` (`2026.09.22-r1`).

Önceki tema, caption/veri koruma ve kıyafet tanıma düzeltmeleri korunarak dört özellik eklendi. Bu dosya güncel kullanım kılavuzudur. `FINAL_*`, `TEMA_*` ve önceki `KIYAFET_*` belgeleri önceki teslimlerin tarihsel kayıtlarıdır; eski patch dosyalarını yeniden uygulamayın.

## 1. Mevcut programı güncelleme

1. Programı, video/caption işlerini ve önceki kurulum pencerelerini kapatın. Mevcut proje klasörünü yedekleyin.
2. **`LoRA-Harvester-Studio-Guncelleme.zip`** içeriğini mevcut **`main.py` dosyasının bulunduğu klasöre** çıkarın. Aynı adlı kaynak dosyalarını değiştirin; klasörleri birleştirin. Eski `src` klasörünü bütünüyle silmeyin.
3. `run.bat` ile açın. Çalışan bir kurulumda bu kaynak güncellemesi için PyTorch veya Ollama modelini yeniden kurmak gerekmez.

Güncelleme ZIP'i `config`, `data`, `models`, `output`, `venv`, `.venv` veya `kohya_ss` içermez. Bu klasörlerinizdeki ayarlar, kıyafet kütüphanesi, modeller ve kullanıcı verileri paket tarafından değiştirilmez. ZIP'i çıkarmak kurulum veya indirme başlatmaz.

**Tam proje:** `LoRA-Harvester-Studio.zip` yeni bir klasöre çıkarılmalıdır. Üst klasörü `LoRA-Harvester-main` adındadır; varsayılan `config` dosyaları içerir. Mevcut kişisel ayarların üzerine tam projenin varsayılanlarını kopyalamayın. Paket bir EXE veya taşınabilir model/sanal ortam arşivi değildir.

## 2. Tek kurulum / onarım sihirbazı

### Artık hangi BAT?

**Kurulum ve onarım için yalnız `install.bat`; programı açmak için `run.bat`.**

`install.bat` artık donanımı otomatik seçip temel program ile isteğe bağlı Python paketlerini terminalde kurar. Eski GPU ve kıyafet kurulum kısayolları kaldırıldı. Uygulama içindeki bakım düğmesi ayrıntılı kontrol/onarım sihirbazını açmaya devam eder; Ollama modelini otomatik indirmez.

`run.bat`, `venv` eksikse `install.bat` dosyasını çalıştırır. Kurulum donanımı algılar ve temel bileşenleri otomatik ekler.

### Kullanım

Önce **Sistemi kontrol et** düğmesine basın. Bu işlem paket kurmaz. Rapor; proje ortamını, temel modülleri, paket bağımlılıklarını, PyTorch CUDA erişimini ve küçük bir CUDA bellek işlemini, ONNX sağlayıcı listesini ve yerel Ollama model listesini denetler. ONNX sağlayıcısının listelenmesi, gerçek WD14 çıkarımının GPU'da başarılı olduğu anlamına gelmez.

Yeni NVIDIA kurulumu için **Temel program + PyTorch GPU desteği** seçin. Bu teslimin kontrollü GPU kanalı **cu124**'tür. Kıyafet modeli henüz yoksa **Kıyafet modeli** seçeneğini de işaretleyin. Mevcut GPU çalışıyorsa GPU kutusunu sırf güncelleme yapılıyor diye seçmeyin: açıkça seçilen GPU onarımı mevcut PyTorch paketlerini seçilen sürümle değiştirir.

ONNX/WD14 için **Mevcudu koru**, **CPU onar** veya **CUDA onar** vardır. Yeni temel kurulum, aksi açıkça seçilmezse CPU ONNX kurar. CUDA onarımı bu pakette `onnxruntime-gpu==1.20.1` ile CUDA 12 PyTorch gerektirir. CUDA 11.8 ile bu ONNX GPU profili birlikte seçilirse işlem reddedilir; CPU ONNX kullanılabilir. CPU/GPU ONNX dağıtımlarının aynı import adını paylaşması nedeniyle onarımda yalnız bir tanesi bırakılır; indirme başarısızsa eski runtime kaldırılmaz.

**Yeni ortam oluşturma profili:** 64-bit Python 3.10, 3.11 veya 3.12. Mevcut farklı sürümlü, çalışır ve yalıtılmış bir ortamın programı açması yalnız bu sürüm aralığı nedeniyle engellenmez. Ancak bu kurulum profilini desteklenmeyen ortamda paket onarımı için kullanmak yeniden ortam oluşturmayı gerektirebilir. `venv` taşınabilir bir klasör kabul edilmez.

Bozuk ortam kendiliğinden silinmez. **Yedekleyerek yeniden oluştur** açıkça seçilirse eski ortam `venv-backup-...` olarak saklanır. Dataset, profil, model ve ayar klasörleri bu işlemde temizlenmez. Kurulum esnasında uygulama açık kalamaz; işletim sistemi kilidi paket değişikliğini engeller. **Settings → Kurulum / onarım** düğmesi aynı sihirbazı açar; paket kurulumu için ana uygulamayı kapatın.

İşlem durdurma, o sırada çalışan pip adımı tamamlandıktan sonra uygulanır. Pip dosya yazarken zorla öldürülmez. Günlükler ve paket/teşhis kayıtları `logs/` altında saklanır. Bütün güncellemelerin otomatik geri alınabildiği iddia edilmez; yarım başarısızlıkta günlük üzerinden onarım gerekir.

Ollama programı eksikse sihirbaz resmi Windows indirme bağlantısını sunar. Ollama'yı kurup açtıktan sonra kıyafet bileşenini çalıştırın. Bu bileşen `qwen3-vl:4b` indirir; ağ ve disk kullanımı için önceden onay istenir. Otomatik yönetici yetkisi veya sessiz servis kurulumu yapılmaz.

**Uyumluluk sınırı:** PyTorch 2.6.0 / torchvision 0.21.0 / torchaudio 2.6.0 seçimi bu eski kod tabanı için açık bir kurulum profilidir; “en yeni sürümler” veya temiz Windows'ta bütünüyle doğrulanmış bağımlılık kilidi olarak sunulmaz. Yeni `requirements-core.txt` ve `requirements-compat.txt` temel/opsiyonel paketleri ayırır. Real-ESRGAN, anime yardımcıları ve InsightFace ayrı seçimdir; özellikle opsiyonel paketlerin Windows derleme/uyumluluğu burada test edilmedi.

## 3. Kıyafet–poz–açı dengeleme

Yer: **Caption Studio → Veri dengesi / Data balance**.

Bu sürüm **caption tabanlıdır**. Görselleri başka bir modele göndererek otomatik poz tahmini yapmaz. Caption'daki kıyafet master tagleri, bilinen poz ve kamera açısı etiketleri sayılır. Eksik bilgi `unknown`, çelişkili bilgi `ambiguous` görünür. Örneğin `standing`, `sitting`, `kneeling`, `from above`, `from side` gibi etiketler değerlendirilir; `looking at viewer` tek başına kamera açısı sayılmaz. Üstten ve yandan görünüm birlikte `side+high` olarak tutulabilir.

Klasörü seçin, tarayın, sonra **Kıyafet / Poz / Açı / Kıyafet × poz × açı** gruplamasından birini seçin. Her gruptan alınacak sayıyı ve seed'i belirleyin. **0 / Auto**, gözlenen uygun grupların en küçüğünü hedefler. Çok sayıda seyrek birleşik grup varsa bu seçenek dataset boyutunu gereğinden fazla azaltabilir; önce yalnız kıyafet veya poz dağılımını inceleyin ve planın seçilen/toplam sayısını kontrol edin.

Seçim her mevcut gruptan deterministik örnekleme yapar. Hedefe ulaşacak kadar örneği olmayan gruplar çoğaltılmaz; eksik kombinasyonlar yaratılmış gibi gösterilmez. `suggested_repeat` rapor bilgisidir, eğitim ayarını otomatik değiştirmez. Bilinmeyenleri dahil et seçeneği kapalıdır; açıldığında unknown ayrı grup olur, doğrulanmış etiket haline gelmez. Ambiguous ve veri çakışmaları otomatik dışa aktarılmaz.

Tablodaki yanlış sınıflandırmayı seçip **Sınıfları kaydet** ile düzeltebilirsiniz. Bu düzeltme yalnız dengeleme metadata'sıdır, `.txt` caption'ı değiştirmez. Kaynak görsel/caption sonradan değişirse düzeltme eski kabul edilir; tekrar kontrol etmek veya kaldırmak gerekir.

**Planı önizle → Ayrı dataset kopyası oluştur** kaynak datasetin dışında yeni bir klasör oluşturur. Görseller, kendi `.txt` / `.json` yan dosyaları ve geçerli kıyafet sahiplik kaydı kopyalanır. Mevcut kaynak görseller/caption'lar silinmez, yeniden adlandırılmaz veya yeniden yazılmaz. Mevcut hedef klasörün üzerine yazılmaz. Hedefin `images` klasörü eğitim girdisidir; kökteki manifest/plan JSON'ları eğitim örneği değildir.

Birebir dosya kopyaları tekilleştirilebilir. Birebir aynı görüntüye farklı caption/sınıflar verilmişse ilk dosya rastgele doğru kabul edilmez, inceleme gerekir. Yakın benzer ama farklı pikselli görüntüler bu yeni panelde birebir kopya sayılmaz. Plan sonrası seçilmiş kaynak dosya veya manuel sınıflar değişirse aktarım reddedilir. Kaynakta kilit/metadata dizini oluşturulabilir; “kaynak görsel ve caption içerikleri değiştirilmez” ile “hiçbir dosya sistemi işlemi yapılmaz” aynı şey değildir.

## 4. Kullanıcı düzeltmelerinden yararlanan kıyafet eşleştirme

Yer: **Caption Studio → Kıyafetler → Analiz ve önizleme → Düzelt / hatırla**.

Görselde ana kişiyi/kıyafeti çerçeveleyin. Doğru profili seçin ve **yalnız görünen parça etiketlerini** işaretleyin. Master tag profilin literal yazımıyla korunur. “Kayıtlı kıyafetlerden hiçbiri” de kaydedilebilir. İsteğe bağlı not, modelin hangi ayrıntıyı karıştırdığını açıklar.

Kaydetmek yalnız düzeltme hafızası ve yeni önizleme üretir. Caption'ı değiştirmek için önceki **İşaretlilere uygula** onayı hâlâ gerekir; yazma ve geri alma eski güvenli günlük mekanizmasını kullanır. Bir yanlış master daha önce caption'a yazılmışsa **“hiçbiri” düzeltmesi onu kendiliğinden silmez**; ilgili önceki uygulamayı geri alın veya caption'ı kontrol ederek düzenleyin.

Hafıza iki biçimde kullanılır:

- **Aynı görsel içeriği + aynı seçim alanı + aynı etkin kıyafet kütüphanesi:** kullanıcı onayı doğrudan yeniden kullanılabilir. Tamamı bu tür kayıtlardan oluşan bir batch Ollama'ya bağlanmak zorunda değildir.
- **Farklı görseller:** açıkça izin verdiğiniz düzeltmeler, ilgili profile ait en fazla iki pozitif/negatif örnek olarak görsel modele eklenir. Hafif görsel benzerlik yalnız bu örnekleri sıralamak içindir; kıyafet kimliğine veya görünmeyen parçalara tek başına karar vermez.

**Bu ağırlık eğitimi/fine-tuning değildir.** Modelin her tahmini kendiliğinden öğrenilmez. Kullanıcı onaylı örneklerin gerçek doğruluğu artırdığı bu ortamda ölçülmedi. Benzer örneklerin görünmeyen parçaları hedef caption'a taşınmamalıdır; mevcut renk varyantı, kişi seçimi ve görünür parça kuralları korunur.

Kütüphane/profil/referans değişikliği veya kaynak içeriği değişikliği eski kaydın kullanımını sınırlayabilir. Bellek değiştiğinde açık önizlemeler de yeniden oluşturulmalıdır. **Bu seçim düzeltmesini unut** kaydı pasifleştirir; caption dosyasını değiştirmez. Bu kayıtlar `data/clothing/feedback/` altında saklanır; kütüphaneyi yedeklerken dahil edin. Arşivlenen eski kayıtlar bir “silinen fotoğraf kurtarma” sistemi değildir; yalnız düzeltme geçmişidir.

## 5. Sabit koşullarda LoRA / checkpoint karşılaştırması

Yer: **Caption Studio → Checkpoint karşılaştır / Compare checkpoints**.

Bu özellik Harvester içine ikinci bir diffusion motoru indirmez. **Yerel Forge/AUTOMATIC1111 servisinin API'sini** kullanır. Kendi WebUI başlatma ayarınıza `--api` ekleyin ve servisi çalıştırın. Checkpoint ve LoRA dosyaları o servisin model listesinde bulunmalıdır. Harvester'da eğitim çıktı klasörünü seçmek tek başına dosyaları WebUI'ye kurmaz; modellerin serviste görünür olması gerekir.

Adresi varsayılan `http://127.0.0.1:7860` olarak bırakıp **Bağlan / modelleri yenile** düğmesine basın. Yalnız localhost HTTP kabul edilir; uzak sunucuya dosya/model yükleme ve kimlik bilgisi saklama yoktur.

**LoRA modunda:** 1–16 aday ve tek sabit ana model seçin. İsterseniz LoRA olmadan ana model sütununu da üretin. **Tam checkpoint modunda:** seçilen ana modeller karşılaştırılır; LoRA uygulanmaz.

Her satıra ayrı test promptu yazın; aynı negatif prompt, seed listesi, sampler, scheduler, CFG, step, çözünürlük, clip skip, VAE ve LoRA ağırlığı tüm adaylara uygulanır. Prompt içindeki ayrıca yazılmış LoRA/extra-network direktifleri reddedilir. Seed `-1` rastgeleliği kabul edilmez. Çözünürlük 64'ün katı olmalıdır. Görsel sayısı, aday × prompt × seed çarpımıdır; başlatmadan önce onay gösterilir.

`Automatic` VAE, farklı checkpointlerde farklı gerçek VAE çözebilir. Daha kontrollü bir karşılaştırma için WebUI'de kurulu aynı VAE adını açıkça seçin. Bu seçim modeller arası mimari uyumsuzluğu çözmez. Illustrious/SDXL LoRA'yı uyumlu ana modelle kullanmak gerekir.

İsteklerde WebUI'nin kalıcı ayarlarını POST ile değiştirmek yerine istek bazlı override ve işlem sonrası geri yükleme kullanılır. WebUI'nin otomatik ek LoRA ayarı kapatılır. Bununla birlikte eklentiler ve başka istemciler Harvester'ın kontrolü dışındadır: karşılaştırma sırasında WebUI'de başka iş başlatmayın, eklenti/ayar/model değiştirmeyin. Aynı Harvester servis adresine eşzamanlı ikinci karşılaştırma kilitle engellenir; başka bir programın WebUI kullanımını bu kilit durduramaz.

Her çıktı PNG olarak saklanır; istek/gerçek metadata, görüntü hash'i, model kataloğu ve deney ayarları `comparison.json` içine yazılır. **Yan yana raporu aç** sabit HTML karşılaştırmasını açar. Önceki sonuçlar panelden yeniden açılabilir. Model kalitesine otomatik puan verilmez veya bir epoch “en iyi” ilan edilmez.

Gelen seed/ölçü ve mevcut metadata'daki step/CFG/sampler/clip skip uyuşmazlıkları hata üretir. Katalog hash'i ile dönen checkpoint hash'i karşılaştırılır. Beklenmeyen LoRA veya bildirilen network hatası reddedilir. Sunucu LoRA hash'i/ilgili doğrulama alanını bildirmiyorsa **doğrulanmadı** uyarısı korunur; sırf PNG döndü diye LoRA'nın başarıyla uygulandığı söylenmez. Metadata kontrolü, sunucunun model hesabını bağımsız olarak doğrulamaz.

Durdurma mevcut örnek bitince etkili olur; başka WebUI işini yanlışlıkla kesebilecek global interrupt gönderilmez. Zaman aşımından sonra aynı istek otomatik tekrar edilmez; sunucu arka tarafta o isteği tamamlıyor olabilir. Tamamlanmış yerel çıktılar hata/iptalde korunur. Aynı seed/ayar farklı donanım, WebUI sürümü, eklenti veya model matematiğinde piksel eşitliği garantilemez.

## 6. Dosyalar ve terminal kullanımı

| Konum | İçerik |
|---|---|
| `logs/` | Kurulum çıktıları, paket listesi ve teşhis raporları |
| `data/clothing/feedback/` | Kullanıcı düzeltmeleri ve onaylı örnek kırpımları |
| `dataset/.lh-dataset/balance_overrides.json` | Görsel/caption hash'ine bağlı manuel dengeleme sınıfları |
| Seçilen dengeleme çıktı klasörü | `images/`, `balance_plan.json`, `manifest.json` |
| `data/evaluations/` veya seçtiğiniz çıktı klasörü | Checkpoint örnekleri, `comparison.json`, `index.html` |

Proje kökünde, çalışan venv Python'uyla:

```powershell
# Salt-okunur kontrol; paket kurmaz.
venv\Scripts\python.exe scripts\setup_wizard.py --check

# Caption dağılımını analiz et; çıktı dataset oluşturmaz.
venv\Scripts\python.exe scripts\dataset_balance_cli.py "D:\dataset" --dimension outfit

# Yeni plan dosyası ve ayrı dataset kopyası oluştur.
venv\Scripts\python.exe scripts\dataset_balance_cli.py "D:\dataset" --dimension pose --per-group 30 --report "D:\balance-plan.json" --export "D:\balanced-new"

# Yerel WebUI katalog bilgisi.
venv\Scripts\python.exe scripts\compare_checkpoints.py --catalog

# Önceki deneyin config bölümünü yeniden doğrula; üretim yapmaz.
venv\Scripts\python.exe scripts\compare_checkpoints.py --recipe "D:\comparison-old\comparison.json"

# Aynı reçeteyle yeni sonuç klasörü oluşturmak için açık üretim izni.
venv\Scripts\python.exe scripts\compare_checkpoints.py --recipe "D:\comparison-old\comparison.json" --run --output "D:\comparisons"
```

Rapor ve dışa aktarım örneklerindeki hedefler yeni olmalıdır. Yolları kendi makinenizdeki gerçek konumlarla değiştirin. `.lh-*` metadata dizinleri eğitim görüntüsü klasörleri değildir; programın dataset tarayıcısı bunları atlar.

## 7. Test sonucu ve doğrulanmayanlar

**285 test geçti; 14 test PyQt5 bulunmadığı için atlandı; başarısız test yok.** Ayrıca 576 alt-kontrol geçti; bunlar 285 sayısına eklenen ayrı üst düzey testler değildir. Önceki 182 testin regresyonları korunmuştur. Tam paket temiz klasöre çıkarılıp yeniden test edildi; ayrıca önceki Final sürümün üzerine güncelleme uygulanarak aynı testler tekrar geçti. Kullanıcı config/data/models/venv kontrol dosyaları değişmeden kaldı. Tarihsel test ve paket kontrol kayıtları orijinal arşivdedir.

Yeni testler gerçek geçici dosya/PNG/JSON işlemleri, dengeli deterministik seçim, kaynak koruma, eski plan reddi, kullanıcı hafızası/unutma, çevrimdışı kesin eşleşme, mevcut caption günlüğüne bağlanma, OS kilidi, proje Python doğrulaması, yanlış GPU bileşen seçimi, CPU→GPU wheel onarımı sözleşmesi ve **gerçek localhost HTTP sunucusu üzerinden taklit WebUI yanıtlarını** kapsar. Paket kurma komutları testlerde taklittir; sisteminize ait veya konteynerin global Python'una test için paket kurulmaz.

**Doğrulanmayanlar:** Windows'ta temiz kurulum, Tk/PyQt pencerelerinin gerçek çizimi ve tema akıcılığı, NVIDIA GPU/VRAM performansı, gerçek Ollama kıyafet doğruluğu ve gerçek Forge/AUTOMATIC1111 diffusion üretimi. Model ağırlıkları ve kullanıcıya ait gerçek anime/2.5D kıyafet örnekleri bu testlere dahil değildir. Bu teslim “bütün olası hataları bitmiş program” garantisi vermez.

Testleri tekrar çalıştırmak için mevcut proje ortamında `python run_tests.py -q -ra` kullanılabilir; test bağımlılıkları `requirements-dev.txt` içindedir.

## 8. Teknik dayanaklar

Aşağıdaki resmi kaynaklar API/sürüm tercihlerini destekler; bu proje için gerçek donanım testi yerine geçmez. Erişim: 22 Eylül 2026.

- Python venv: https://docs.python.org/3/library/venv.html
- PyTorch 2.6.0 / torchvision 0.21.0 CUDA wheel eşleşmeleri: https://pytorch.org/get-started/previous-versions/
- ONNX Runtime CUDA/cuDNN matrisi: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- WebUI API: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API
- WebUI gerçek işleme alanları / dönen metadata: https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/modules/processing.py
- Varsayılan LoRA ve LoRA hash davranışı: https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/extensions-builtin/Lora/extra_networks_lora.py
- Seed ve yeniden üretilebilirlik sınırları: https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds
- Ollama görsel generate API: https://docs.ollama.com/api/generate

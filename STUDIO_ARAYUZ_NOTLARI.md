# LoRA-Harvester — Studio masaüstü arayüzü ve akıllı öneriler

**Sürüm:** `2026.09.22-r4`\
**Temel:** Bu konuşmada teslim edilen `LoRA-Harvester-Senkronizasyon.zip` (`r3`).

Yeni ana pencere, seçilen açık/nötr ve mor vurgulu stüdyo tasarımını gerçek program özellikleriyle birleştirir. Bu teslim yalnız bir tasarım görseli veya görev planı değildir: çalıştırma girişine bağlanmış Python/PyQt5 kaynak kodu içerir. Eski backend ve çalışan araç denetleyicileri yeniden kullanılmıştır. **Windows'ta tam açılış, gerçek Qt çizimi ve GPU performansı bu ortamda doğrulanmadı.**

## 1. Çalışan kurulumun güncellenmesi

1. Uygulamayı, video/caption/eğitim işlerini kapat. Mevcut proje klasörünü yedekle.
2. **`LoRA-Harvester-Studio-Arayuz-Guncelleme.zip`** içeriğini mevcut **`main.py` dosyasının bulunduğu klasöre** çıkar; aynı adlı dosyaları değiştir ve klasörleri birleştir. Eski `src` klasörünü bütünüyle silme.
3. **`run.bat`** ile aç. Çalışan kurulumda bu kaynak güncellemesi için Python, PyTorch veya Ollama modellerini yeniden kurmak gerekmez.

Güncelleme ZIP'i `config`, `data`, `models`, `output`, `venv`, `.venv`, `kohya_ss` içermez. Kişisel ayarlar, kıyafet profilleri, referanslar, modeller ve sanal ortam bu pakette yoktur; üzerine varsayılan veri yazılmaz. ZIP'i çıkarmak kendiliğinden kurulum, indirme veya veri seti işlemi başlatmaz.

**Tam proje:** `LoRA-Harvester-Studio-Arayuz.zip` içindeki `LoRA-Harvester-main` klasörünü yeni bir konuma çıkar. Yeni kurulumda `install.bat`, ardından `run.bat` kullan. Bu ZIP varsayılan `config/config.yaml` içerir; çalışan kurulumun kişisel ayarlarını bununla değiştirme. Model ağırlıkları ve kişisel veri setleri ZIP'e dahil değildir. Sanal ortam klasörünü başka yere kopyalamak yerine yeni ortamı kur.

**Eski arayüze dönüş:** Aynı klasörde **`run_classic.bat`** çalıştır. Aynı veri, model ve backend kullanılır; yalnız eski ana arayüz açılır. Tercihler uygulama genelindedir; bu seçenek geçmiş bir sürüme veri geri alma işlemi yapmaz.

## 2. Yeni ekran düzeni

Üst çubuk gerçek çalışma klasörünü, dosya/tag aramasını, açık/koyu tema düğmesini ve öneriler erişimini gösterir. Sahte proje hesapları, avatar, abonelik rozeti, sohbet modeli veya temsili başarı puanı eklenmedi.

Sol menü:

| Bölüm | Bağlandığı gerçek özellik |
|---|---|
| Genel bakış | Çalışma klasörü ve veri seti tarama kontrolleri |
| Kütüphane | Kaydedilmiş dosyaların aranabilir/filtrelenebilir tablosu, önizleme, Edit/Clothing erişimi |
| Veri düzenleme | Mevcut Edit denetleyicisi; tek caption/taslak deposu |
| Otomatik etiketleme | Mevcut caption üretim araçları ve ayarları |
| Kıyafet eşleştirme | Mevcut profil kütüphanesi, referanslar, ana kişi seçimi, analiz/önizleme/uygula/geri al |
| Veri dengesi | Mevcut caption tabanlı kıyafet/poz/açı dengeleme |
| Karşılaştırma | Mevcut yerel Forge/AUTOMATIC1111 checkpoint/LoRA karşılaştırması |
| Dışa aktar | Mevcut Review Grid / Kohya export iş akışı |
| Diğer araçlar | Videodan görsel çıkarma, karakter sıralama, upscale, etiket sıklığı, caption kalite denetimi, görsel eleme ve Kohya eğitimi |
| Ayarlar | Mevcut gerçek ayarlar ve kurulum/onarım; açılır-kapanır bölümler |

Ayarların gizlenmesi değerlerini değiştirmez veya özelliği kapatmaz. Eski araçların ayrıntılı formları ve işlevleri korunur; her form baştan yazılmış veya taslak görselle piksel piksel aynı hale getirilmiş değildir. Yeni kabuk, yeni kütüphane ve Edit yerleşimi bu teslimin ana arayüz değişiklikleridir.

### Edit çalışma alanı

Büyük görsel önizlemesi, yanında gerçek caption metni ve tag düzenleme; altta yatay görsel şeridi bulunur. Kaydet, sonraki görsele geçerek kaydet ve geri al mevcut güvenli işleyicilere bağlıdır. Eski toplu tag ekle/sil/değiştir ve tümünü kaydet komutları **Toplu işlemler** menüsündedir.

- `Ctrl+O`: Veri seti aç.
- `Ctrl+K`: Dosya/tag aramasına geç.
- Edit içindeyken `Ctrl+S`: Seçili caption'ı kaydet.
- Edit içindeyken `Ctrl+Shift+S`: Tüm düzenlemeleri kaydet.
- Edit içindeyken `Alt+Sol / Alt+Sağ`: Önceki/sonraki görsel.

**Caption taslağı ile kaydedilmiş dosya ayrı tutulur.** Akıllı öneriler ve kütüphane kaydedilmiş dosyaları inceler. Kaydetmediğin metin dosyaya otomatik yazılmış sayılmaz. Edit–Clothing kaydedilmiş caption senkronizasyonu korunur; klasör değişiminde taslak koruma onayı atlanmaz. Kütüphaneden farklı Clothing görseline geçmek diğer mevcut analizleri ve önizlemeleri sıfırlamaz.

### Tema ve pencere

Yeni ve tercih dosyası olmayan kurulumda koyu tema, mor vurgu ve İngilizce başlar. Kurulum/onarım sihirbazı da İngilizcedir. **Mevcut tercih dosyası varsa seçtiğin renk, mod ve dil korunur.** Üstteki **Açık mod / Koyu mod** düğmesi temayı değiştirir. Kütüphane, veri dengesi, kıyafet analizi, caption denetimi, kıyafet referansları ve karşılaştırma listelerinde görsel adının yanında küçük önizleme gösterilir; yalnız görünür satırların görselleri arka planda yüklenir.

Öneriler, yeterince geniş pencerede Genel Bakış/Kütüphane/Edit yanında açılır. Dar pencerede aynı **Öneriler** düğmesi tam sayfa öneri ekranına götürür; özellik kaybolmaz. Teknik formlar ihtiyaç olduğunda kaydırılabilir. Gerçek Windows ölçekleme, farklı DPI ve çoklu monitör davranışı yerinde test edilmelidir.

## 3. Akıllı öneriler: gerçek bulgular, otomatik müdahale yok

**Genel bakış → Veri setini tara** ile başlat. Klasör Edit tarafından kabul edildiğinde ilk tarama da sıraya alınır. İstersen alt klasörleri kapat, kısa kenar sınırını değiştir veya SHA256 kopya taramasını aç. Tarama iptal edilebilir.

| Kontrol | Gerçekte ölçülen |
|---|---|
| Eksik caption | Görselin yanında `.txt` bulunmaması |
| Boş caption | Dosyanın var olup boş/yalnız boşluk içermesi |
| Okunamayan caption | İzin, kodlama, dosya türü, sınır veya tarama sırasında değişim sorunu |
| Caption ad çakışması | Aynı klasörde aynı kök ada sahip görsellerin aynı `.txt` dosyasını paylaşması |
| Görsel başlığı | Görsel format başlığının ve boyutunun okunabilmesi; tam piksel bütünlüğü denetimi değildir |
| Küçük kısa kenar | Seçilen piksel eşiğinin altında boyut; varsayılan 512, 0 ile kapatılır |
| Tekrar tagler | Büyük/küçük harf ve boşluk/alt çizgi normalizasyonuyla yinelenen caption etiketleri |
| Birden fazla kıyafet master | Etkin kütüphaneden birden fazla master etiketinin aynı caption'da bulunması |
| Birebir dosya kopyaları | İsteğe bağlı SHA256 byte eşitliği; tüm kopya grubu ve fazladan kopya sayısı ayrı |
| Seyrek kıyafet grupları | Caption'da gözlenen bir grubun, en büyük gözlenen grubun dörtte birinden az örneği olması |

Bu motor **bulanıklık/estetik puanı, gerçek kıyafet doğruluğu, yakın benzer poz, yüz/kimlik veya eğitim başarısı tahmin etmez**. Görünmeyen parçaları çıkarmaz. Model yüklemez; mevcut caption etiketlerinden ve dosya özelliklerinden hareket eder. Görülmeyen bir kıyafet profilini zorunlu eksik sınıf saymaz.

Üst sayaçlar: taranan görsel sayısı, dolu caption sayısı, eksik/boş caption sayısı, tarandıysa fazladan birebir kopyalar. Dolu caption, doğru caption garantisi değildir. SHA256 kapalıysa kopyalar için **Taranmadı** yazar; sıfırmış gibi gösterilmez. Sahte **87/100 sağlık**, **eğitime hazır** veya ölçülmemiş doğruluk yüzdesi yoktur.

Öneride **İlgili görselleri göster** ilgili dosyaları kütüphanede filtreler. Buradan görsel Edit veya Clothing'de açılabilir. Hiçbir öneri tıklaması kendiliğinden görsel silmez, caption yazmaz, eğitim başlatmaz veya ücretli/bulut model çağırmaz. Denge önerisi de önce ilgili görüntüleri gösterir; veri kopyası üretmek için mevcut Dengeleme aracındaki açık işlem gerekir.

Kaydet/Clothing uygula/geri al sonrası eski sayaçlar geçersizleşir; kaynaklar boşta kalınca tarama birleştirilerek yenilenir. Önceki klasöre ait geç gelen tarama sonucu yeni klasörün üzerine yazılmaz. **Harici programda dosya düzenlemek veya kıyafet profillerini değiştirmek, her durumda otomatik dosya sistemi izleyicisiyle algılanmaz; bu durumda Taramayı yenile kullan.** Kütüphane bir tarama anı görüntüsüdür.

## 4. Performans değişiklikleri ve sınırlar

Edit, yeni düzende klasörün bütün görsellerini tek tek tam boyutlu `QPixmap` ile açıp küçültmez. Görünen şerit öğeleri ihtiyaç olduğunda yüklenir; büyük önizleme ayrı istektir. Arka plan çözümlemesi en fazla iki işçi kullanır. Kuyruk 40 istekle, paylaşılan QImage LRU önbelleği 48 MiB / 256 kayıtla sınırlıdır. Bu değerler **uygulamanın toplam RAM kullanım sınırı değildir**: çözümleme tamponları ve ek Qt pixmap'leri ayrıca bellek kullanır.

QPixmap GUI tarafında oluşturulur. Eski önizlemenin geç tamamlanıp başka görselin yerine konması seçim anahtarıyla engellenir. Filmstrip pencere dışındaki ikonları bırakır. Kütüphane, satır başına ayrı QWidget üretmek yerine Qt model/view tablosu kullanır.

Öneri taraması QThread içinde çalışır; ilerleme bildirimleri seyreltilir. `data/studio/metadata.sqlite3` dosyasında stat imzasına bağlı boyut/hash önbelleği kullanılır. Caption içerikleri her taramada tekrar okunur. Cache hatası taramayı durdurmak yerine yavaş ama önbelleksiz yola düşer.

**Kalan maliyetler:** Eski Edit klasör listesi/caption güvenlik snapshot'ları hâlâ ana iş parçacığında hazırlanır; çok büyük veri setinde ilk açılışta gecikme olabilir. Bütün eski araç sayfaları başlangıçta bir kez oluşturulur; tam lazy-page mimarisine geçilmedi. Kıyafet/Ollama model çağrıları, WD14/YOLO/upscale/eğitim hesapları bu arayüz güncellemesiyle hızlandırılmış sayılmaz. Windows'ta ölçülmüş genel hızlanma yüzdesi yoktur.

### Sınırlı benchmark

Linux/Python 3.13.5 üzerinde 260 **sentetik, küçük dosyalı, 768×1024 PNG**; SHA256 açık metadata taraması:

| Geçiş | Süre | Önbellekten gelen görsel |
|---|---:|---:|
| İlk | 0,144 sn | 0 |
| Tekrar | 0,093 sn | 260 |

Bu rakamlar yalnız bu sentetik tarama yükünündür. UI açılışı, gerçek karmaşık anime PNG/JPEG dosyaları, SSD/HDD, RTX 3070 Ti veya model çıkarım performansı ölçümü değildir. Ham benchmark kaydı orijinal arşivdedir.

## 5. Test ve doğrulama

**Son çalışma: 402 test geçti, 28 test atlandı; başarısız test yok.** 597 alt-kontrol ayrıca geçti; bunlar geçen ana test sayısına eklenmez.

Yeni testler gerçek geçici dosyalarda tarama, caption durumları, hash/önbellek, çakışmalar, iptal, kaynak korunması ve gerçek kaynak metotlarının küçük test nesneleriyle rota/senkronizasyon davranışını kapsar. Bunlar bir pencere çizimi testi değildir. Önceki video, kıyafet, geri alma, kurulum, karşılaştırma ve caption senkronizasyon regresyonları tekrar çalıştırıldı.

**Atlananlar:** Ortamda PyQt5 olmadığı için önceki 20 ve yeni 8 gerçek Qt testi çalıştırılamadı. PyQt5 kurma denemesi ağ/DNS hatasıyla başarısız oldu. Bu nedenle gerçek yeni pencere açılışı, tüm widget ölçüleri, canlı tema çizimi, klavye odağı ve Windows/DPI davranışı burada onaylanmış değildir. Yeni Qt testleri desteklenen ortamda çalıştırılabilir; tüm ana pencere/görsel kabul testinin yerine geçmez.

**2026-09-23 Windows ek doğrulaması:** Studio gerçek Windows masaüstünde koyu temayla açıldı; açık/koyu geçişi, 238 görsellik klasör taraması ve güvenli dört görsellik örnekle kütüphane, veri dengesi ve kıyafet listelerindeki önizlemeler görüldü. Altı tema testi ve yeni thumbnail çizim testi geçti. Mevcut Studio Qt test dosyasındaki iki test, paylaşılan QStringListModel nesnesinin silinmesi nedeniyle hâlâ başarısız; bu hata bu tema/önizleme değişikliklerinde oluşmadı. Yukarıdaki Linux/Python 3.13 test sayıları tarihsel kayıttır.

**Ayrıca doğrulanmayanlar:** Temiz Windows kurulumu, CUDA/GPU performansı, gerçek Ollama/Qwen kıyafet tanıma, WD14/Florence/YOLO/Real-ESRGAN çıkarımı ve gerçek Forge üretimi. Testlerde gerektiğinde kontrollü model/HTTP yanıtları kullanılır.

Çalışan proje ortamında:

```powershell
venv\Scripts\python.exe -m pip install -r requirements-dev.txt
venv\Scripts\python.exe run_tests.py -q -ra

# Pencere/model açmadan salt-okunur veri kontrolleri:
venv\Scripts\python.exe scripts\studio_scan.py "D:\dataset"
venv\Scripts\python.exe scripts\studio_scan.py "D:\dataset" --duplicates --report "D:\dataset-report-new.json"
```

Rapor hedefi yeni olmalıdır; mevcut dosyanın üzerine yazılmaz. `--report` yalnız açıkça istenen JSON dosyasını oluşturur; caption'ları değiştirmez.

### Windows'ta kabul kontrolü

Kopya bir veri setinde klasör aç; Edit taslağı oluştur; sekme/rota değiştirip geri dön; kaydet; Clothing önizlemesini kontrol et. Clothing uygula/geri al sonrası Edit'i kontrol et. Eksik caption önerisine tıkla ve doğru dosyaların filtrelendiğini doğrula. İki farklı klasör arasında geçerken eski taramanın yeni sayacı ezmediğini kontrol et. Açık/koyu tema, dar/geniş pencere ve %125/%150 DPI'da görünümü incele. Sorunda aynı klasörde `run_classic.bat` ile mevcut araçlara erişebilirsin.

## 6. Paket kayıtları

Tarihsel sürüm ve test raporları orijinal arşivde korunur; bu pakette tekrar çalıştırılabilir test kaynakları `tests/` altındadır. Eski `.patch` dosyalarını bu teslimin üzerine ayrıca uygulama.

Teknik dayanaklar (Qt sürümünden bağımsız ilke açıklamaları; PyQt5 çalışma testi değildir): Qt resmî `QListView` batched/uniform item size ve QObject iş parçacığı belgeleri: https://doc.qt.io/qt-6/qlistview.html ve https://doc.qt.io/qt-6/threads-qobject.html.

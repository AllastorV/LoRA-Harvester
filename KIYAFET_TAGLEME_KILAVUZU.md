# LoRA-Harvester — Referans tabanlı kıyafet etiketleme

**Sürüm:** Kıyafet modülü 1.0 · **Tarih:** 22 Eylül 2026\
**Temel:** Bu konuşmada teslim edilen `LoRA-Harvester-Tema-Duzeltilmis.zip`.

Bu paket bir görev promptu veya arayüz maketi değil; profil kütüphanesi, yerel görsel model bağlantısı, kişi seçimi, görünür parça kararları, caption önizlemesi, yazma/geri alma ve video/caption iş akışı bağlantılarını içeren kaynak kod güncellemesidir. Önceki tema düzeltmeleri korunmuştur. **Gerçek kıyafet tanıma kalitesi, Windows arayüzü ve GPU performansı bu ortamda doğrulanmadı.** İlk kullanımda küçük bir örnek setiyle önizleme yapılmalıdır.

## 1. Kurulum

1. Harvester Python paketleri için proje kökündeki `install.bat` dosyasını çalıştır.
2. Kıyafet özelliği için Ollama'yı ayrı kur ve aç; ardından `ollama pull qwen3-vl:4b` komutuyla modeli indir.
3. `run.bat` ile uygulamayı aç. **Caption Studio → Kıyafetler / Clothing → Model ve ayarlar** bölümünde **Modeli kontrol et** düğmesine bas.

Eski kıyafet güncelleme ZIP'lerini bu tam paketin üzerine uygulama. `venv`, `models`, `config` ve kişisel çıktıları koru.

### Yerel model

Varsayılan `qwen3-vl:4b`, servis adresi `http://127.0.0.1:11434` olarak ayarlıdır. Gerektiğinde daha küçük `qwen3-vl:2b` arayüzden seçilip indirilebilir. Modelin küçük olması bu görevi yeterli doğrulukla yapacağını garanti etmez. `qwen3-vl:8b` de seçilebilir; bellek gereksinimi kendi bilgisayarında kontrol edilmelidir. **RTX 3070 Ti üzerinde hız/VRAM ölçümü yapılmadı.**

Kurulum betiği ve yeni modül mevcut Python, PyTorch, transformers veya WD14 paketlerini yükseltmez. Görsel çıkarım ayrı Ollama sürecinde yapılır. Model dosyaları ZIP'e dahil değildir; Ollama'nın kendi depolama alanına indirilir. Model indirmek internet gerektirir. Çıkarım bağlantısı yalnız yerel HTTP adreslerine izin verir; model adındaki `cloud` ve `/api/show` yanıtındaki uzak model alanları reddedilir. Görselleri barındıran bir bulut servisi bu kodda kullanılmaz.

## 2. Kıyafet profilini oluşturma

**Kıyafet kütüphanesi → Yeni** ile başla. Kıyafete bir ad ver, tam promptunu yapıştır ve **Promptu parçalara ayır** düğmesine bas:

```text
shhooldress, serafaku, blue skirt, black thighighhs
```

İlk tag `shhooldress` master olur. Diğer tagler sırayla parça tablosuna aktarılır. Yazım otomatik düzeltilmez: `serafaku` ve `black thighighhs` gibi özel yazılmış tagler çıktı sırasında aynen korunur. Parça tablosundaki **Görsel anlamı** alanı, modele yardımcı olan açıklamadır; caption'a yazılan ayrı bir tag değildir.

Bir veya birden fazla referans ekle. Referans çok kişiliyse **Alan seç** ile yalnız ilgili kişiyi/kıyafeti çevrele. **Referansları analiz et** mevcut profili kaydeder, referanslardaki görünen ayrıntılar için açıklama önerileri getirir. Önerileri kontrol et ve **Profili kaydet** ile son halini kaydet. Dolu açıklamaların üstüne otomatik yazılmaz. Profil değiştirirken veya uygulamayı kapatmadan önce düzenlemelerini kaydet; profil editörü otomatik kaydetmez.

Aynı tasarımın kırmızı ve mavi versiyonlarını **ayrı profiller ve ayrı master taglerle** tanımla. İsteğe bağlı Grup alanı düzenleme içindir; farklı renkleri tek kıyafet haline getirmez. Master tagler kütüphane içinde benzersiz olmalıdır. Sekiz profil sınırı yoktur; testlerde 12 etkin profil birlikte ele alınmıştır. Bir profilde 1–64 parça tagi desteklenir.

Referans seçiminde yüz ve poz benzerliği yerine kıyafetin rengi, kesimi ve ayırt edici ayrıntılarını gösteren örnekler kullan. Farklı açıları eklemek faydalı olabilir; çok fazla referans ilk analiz süresini artırır. Programın yanında hazır kıyafet referansları veya sana özel eğitilmiş ağırlıklar gelmez: bu konuşmada henüz kıyafet örnekleri verilmedi.

## 3. Görselleri analiz etme

**Analiz ve önizleme** bölümünde klasör veya görseller seç. Alt klasörleri tarama isteğe bağlıdır. Bir görseli seçerek sağdaki önizlemede **ana kişiyi/kıyafeti sürükleyerek çevreleyebilirsin**. Bu kutu sadece analiz alanıdır; kaynak görseli kesmez, üzerine çizmez veya yeniden kaydetmez.

Manuel kutu yoksa varsayılan en büyük kişi seçilir. Ayarlarda merkeze en yakın kişi veya yalnız manuel seçim kullanılabilir. Çok kişili görselin bütün kıyafetleri birleştirilmez. İnsanlar örtüşüyorsa veya güvenilir kişi seçimi yapılamıyorsa sonuç incelemeye ayrılır. Manuel seçimin kendisinde başka insanların kıyafetleri kalıyorsa alanı daralt.

**Seçileni analiz et / Tümünü analiz et** yalnız önizleme üretir. Tabloda Eşleşti, İnceleme, Eşleşmedi veya Hata durumu; sağda eski/yeni caption ve karar ayrıntıları gösterilir. Kabul edilen sonuçları kontrol edip **İşaretlilere uygula** ile dosyaya yaz. Bu işlem ayrıca onay sorar. **Son uygulamayı geri al** son yazma grubunu, sonradan yapılan düzenlemeleri ezmeden geri almaya çalışır.

Edit sekmesinde kaydedilmemiş caption değişiklikleri varsa önce bunları kaydet veya geri al. Kıyafet uygulama/geri alma işlemi bu değişiklikleri sessizce silmek yerine engellenir.

## 4. Karar ve çıktı kuralları

Aşağıdaki ilk üç satırda kıyafetin kimliğinin yeterince doğrulandığı varsayılmıştır:

| Hedefte görünenler | Yeni kıyafet tagleri |
|---|---|
| Üst, mavi etek, siyah çorap | `shhooldress, serafaku, blue skirt, black thighighhs` |
| Üst ve mavi etek | `shhooldress, serafaku, blue skirt` |
| Yalnız ayırt edilebilir üst | `shhooldress, serafaku` |
| Yalnız yüz veya tanınmayan giysi | Yazılmaz |
| İki profil de görünür bölüme uyuyor | İnceleme; master tahmin edilerek yazılmaz |

Master tag kabul edilen eşleşmede başta kalır, arkasından yalnız doğrulanan parçalar kullanıcı sırasıyla gelir. Bir parça taginin renk gibi bütün nitelikleri görüntüden seçilebilir olmalıdır. Parçanın tamamının kadrajda olması gerekmez; tanınabilir bir bölüm yeterli olabilir. Gizli/örtülü parça, referansta bulunduğu için hedefe eklenmez.

İki renk varyantını ayıran tek parça kadraj dışındaysa sistem skor farkına güvenerek birini seçmez. Birden fazla uyumlu adayda inceleme sonucu verir. **Bu yaklaşım yanlış master eklemeyi azaltmayı amaçlar; gerçek model hatalarını tamamen engellediği doğrulanmış değildir.** Kimlik ve parça skorları modelin tahminleridir, ölçülmüş doğruluk yüzdeleri değildir. Varsayılan eşikler sırasıyla 0.85 ve 0.80'dir.

### Mevcut caption'ın korunması

Birleştirme sırası `master, görünen parçalar, mevcut diğer etiketler` şeklindedir. Tekrar kontrolünde büyük/küçük harf ve boşluk/alt çizgi farkları normalize edilir; çıktıdaki profil taglerinin yazımı korunur.

**Önceden WD14'ün veya senin yazdığın görünmeyen bir kıyafet tagi kendiliğinden silinmez.** Sistem yalnız kendi önceki işlemlerinde eklediği tagleri sahiplenir ve gerekirse değiştirir. Önceden var olan doğrulanmamış parça tagi korunuyorsa önizlemede uyarı gösterilir. Bu ayrım, elle yazılmış bilgilerin yanlış bir model kararı yüzünden kaybolmasını önler; bütün caption'ın sıfırdan görünürlük denetimi yapıldığı anlamına gelmez.

Yalnız görselin yanındaki **`.txt`** değiştirilir. Daha önce üretilmiş WD14 `.json` metadata dosyaları bu işlemle güncellenmez. Sonradan başka bir işlemle caption'ı tamamen yeniden üretmek kıyafet taglerini değiştirebilir; korunmasını istiyorsan yeniden kıyafet geçişi yap veya aşağıdaki otomatik caption-sonrası seçeneğini aç.

## 5. Video ve caption sonrasında otomatik çalışma

Model ve ayarlarda iki bağımsız seçenek bulunur:

- **Video çıkarımı bittikten sonra otomatik uygula:** bu çalıştırmada kaydedilmiş bütün son kırpımları analiz eder; yalnız arayüze gösterilen seyrek önizleme kareleriyle sınırlı değildir.
- **Caption üretimi bittikten sonra otomatik uygula:** seçili görsel klasörünü, mevcut alt klasör ayarını izleyerek analiz eder. Caption üreticisinin mevcut `.txt` nedeniyle atlamış olduğu görseller de bu son geçişe dahil olabilir.

**İkisi de varsayılan olarak kapalıdır.** Açıp kaydetmek, kabul edilen sonuçların her iş sonunda ayrı önizleme onayı olmadan yazılmasına izin verir. Yedek ve geri alma kayıtları yine oluşturulur. Önce birkaç gerçek görselde manuel önizleme ile ayarları doğrula.

Bu geçiş, önceki video/caption modellerinin temizliği tamamlandıktan sonra çalışır. Son kaydedilen kırpım kullanıldığı için kaynak video karesinde olup kırpımda görünmeyen parça, sırf kaynakta vardı diye eklenmez. Kıyafet işlemi hata verirse önceden çıkarılmış görseller veya üretilmiş caption'lar bu yüzden silinmez. Durdurmadan önce tamamlanmış yazmalar korunur ve geri alınabilir.

## 6. Hız, bellek ve yaklaşımın sınırları

İlk taslaktaki embedding/top-k ön elemesi yerine bu sürümde **bütün etkin profiller, bütün kayıtlı referanslarıyla yerel görsel model üzerinden karşılaştırılır**. Amaç, birbirine çok benzeyen bir renk varyantının ön elemede elenip yanlış master'ın tek aday kalmasını önlemektir. CLIP/WD14 benzerliğiyle kıyafet kimliği varmış gibi sunulmaz.

Bunun bedeli ilk geçişin daha pahalı olmasıdır: bir görsel için varsayılan akışta bir kişi konumlandırma isteği ve her etkin profil/referans çifti için bir karşılaştırma isteği yapılır. Manuel kişi kutusu ilk isteği atlar. Gereksiz referansları azaltmak veya o çalışma için ilgisiz profilleri pasifleştirmek işi azaltabilir. **Gerçek görsel başına süre ya da hızlanma yüzdesi ölçülmedi.**

Önbellek doğrulanmış model yanıtlarını tutar; görüntü, referans, model özeti, prompt/şema veya ilgili ayarlar değiştiğinde farklı anahtar kullanılır. Arayüzün model çıkarımı ana GUI iş parçacığına yüklenmez. Çok büyük görsellerin okunması/liste yüklemesi ve işletim sistemi kaynak baskısı yine gecikme yaratabilir; tüm arayüzün hiç takılmayacağı iddia edilmez.

Model bağlamı varsayılan 8192, analiz görüntüsünün uzun kenarı 768, istek zaman aşımı 180 saniyedir. Bunlar kalite/hız dengesi için başlangıç ayarlarıdır. Çok uzun profil açıklamalarında bağlam sınırı uyarısı alınırsa açıklamaları kısalt veya bellek durumuna göre bağlamı artır. Çıktı yarıda kesilirse sonuç tahminle tamamlanmaz; hata/inceleme gerekir.

## 7. Kayıtlar, yedek ve güvenlik

| Konum | İçerik |
|---|---|
| `data/clothing/profiles.json` | Profiller, literal tagler, parça açıklamaları |
| `data/clothing/references/` | İçe aktarılmış referans kopyaları |
| `data/clothing/settings.json` | Model ve davranış ayarları |
| `data/clothing/selections.json` | Görsel içeriğine bağlı manuel seçimler |
| `data/clothing/cache.sqlite3` | Yeniden kullanılabilir yerel analiz yanıtları |
| `data/clothing/undo_jobs.json` | Son uygulama gruplarının geri alma kayıtları |
| `görsel_klasörü/.lh-clothing/state/` | Programın eklediği taglerin sahiplik bilgisi |
| `görsel_klasörü/.lh-clothing/history/` | Yazmadan önce oluşturulan caption yedeği ve işlem günlüğü |

Profil/kütüphaneyi taşırken `data/clothing` klasörünü birlikte taşı. Geri alma için ilgili görsel klasörlerinin `.lh-clothing` dizinleri de gereklidir. Bu dizin bir eğitim görseli klasörü değildir; eğitim aracının ilgisiz alt klasörleri taramadığını kontrol et. Önbellek, uygulama kapalıyken kaldırılırsa yeniden oluşturulur; **geri alma kayıtlarını aynı amaçla silme**.

Önizlemeden sonra kaynak görsel, caption, profil veya referans değişirse yazma reddedilir. Aynı klasörde aynı kök ada sahip `a.png` ve `a.jpg`, tek `a.txt`yi paylaşacağı için etiketleme engellenir. Caption yazımı geçici dosya + atomik değiştirme kullanır. Eski bytes, caption değişmeden önce yedeklenir. Uygulama kapansa bile bu günlükten kurtarma yapılabilir. Sonradan değiştirilmiş caption veya kaynak görsel üzerine geri alma zorlanmaz.

Kişisel profiller, referanslar, kaynak görseller ve model ağırlıkları bu teslim paketlerine eklenmemiştir. Program çalışma sırasında bunları senin makinenizde oluşturur.

## 8. Komut satırı ve kurtarma

Projenin mevcut sanal ortamını etkinleştirip kök klasörde çalıştır:

```powershell
python scripts/clothing_cli.py --help
python scripts/clothing_cli.py check
python scripts/clothing_cli.py list

# Yalnız analiz; caption dosyalarına yazmaz. Rapor isteğe bağlıdır.
python scripts/clothing_cli.py analyze "D:\dataset" --recursive --report "outfit_report.json"

# Açık yazma izni: sadece kabul edilen eşleşmeleri uygular.
python scripts/clothing_cli.py analyze "D:\dataset" --recursive --apply --report "outfit_report.json"

# Son tamamlanmış uygulama grubunu geri al.
python scripts/clothing_cli.py undo

# Kesintide gruba kaydedilememiş tek günlük için kurtarma:
python scripts/clothing_cli.py undo-journal "D:\dataset\.lh-clothing\history\GERCEK_GUNLUK_ADI.json"
```

Alternatif kütüphane kökü için global `--library-root` parametresini alt komuttan önce ver. Gerçek günlük adını `history` klasöründen seç. Komutlar bilinmeyen/çakışan bir kaydı sessizce ezmek yerine hata verir. Model indirmek için `python scripts/clothing_cli.py pull` kullanılabilir; bu komut açık bir indirme talebidir.

## 9. Test kapsamı

Teslim öncesi test sonucu: **132 test keşfedildi; 122 geçti, 10 atlandı.** Atlananların 6'sı önceki tema Qt testleri, 4'ü yeni kıyafet widget testleridir. Ortamda PyQt5 bulunmadığı için gerçek pencere oluşturma/çizim yapılmadı. Çalışan 122 testin 31'i önceki tema regresyon testleridir.

Yeni testler profil CRUD ve referans güvenliği, literal tagler, 12 profil, görünür parça/master kararları, renk belirsizliği, ana kişi kutusu, önbellek, varsayılan salt-okunur analiz, yazma çatışmaları, tam bytes geri alma, hata/iptal ve kaynak entegrasyon sırasını kapsar. HTTP testleri **gerçek localhost HTTP sunucusuyla, fakat taklit Ollama yanıtlarıyla** çalışır. Karar testleri sentetik görseller ve kontrollü model çıktıları kullanır. Bunlar modelin gerçek anime kıyafetlerini doğru tanıdığını gösteren bir değerlendirme değildir.

Kontrol komutu:

```powershell
python -m unittest discover -s tests -v
```

Tarihsel test listesi ve dosya hash'leri orijinal arşivdedir. Mevcut dosyalardan yalnız `src/ui/main_window.py` ve `src/ui/caption_studio_page.py` entegrasyon için değiştirilmiştir.

### Kendi kurulumunda son kontrol

Bir tam kıyafet, belden yukarı kırpım, yalnız üst, başka kıyafet, iki kişi ve ayırt edici renk bölümü görünmeyen iki varyant örneğiyle dene. Önce önizlemeyi, sonra birkaç kopya görselde uygula/geri al işlemini kontrol et. Yeni sekme açıkken vurgu rengi ve açık/koyu tema değiştir; yazılmamış profil metninin ve kişi seçiminin korunduğunu doğrula. Model gerçek veri üzerinde yanlış eşleşiyorsa otomatik uygulamayı kapalı tut.

## 10. Kullanılan API belgeleri

- Ollama Generate: https://docs.ollama.com/api/generate
- Yapılandırılmış görsel çıktı: https://docs.ollama.com/capabilities/structured-outputs
- Model indirme: https://docs.ollama.com/api/pull
- Bellekte tutma ve bağlam ayarları: https://docs.ollama.com/faq
- Qwen3-VL model ailesi: https://ollama.com/library/qwen3-vl
- Qwen resmi GGUF model kartı: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF

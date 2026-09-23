# LoRA Harvester — Canlı AI kontrolü

**Teslim:** `2026.09.22-ai1` · **Temel:** `LoRA-Harvester-Genel-Performans-Tam.zip` / `2026.09.22-perf2`.

Bu paket, mevcut Python/PyQt5 masaüstü programına canlı kontrol katmanı ekler. Yeni bir görsel maket, ayrı bir web arayüzü veya görev promptu değildir. Video çıkarma/performance değişiklikleri, Studio arayüzü, kıyafet sistemi, görünür parça/master kuralları ve Edit–Clothing senkronizasyonu korunur.

**Çalışma modeli:** `Codex / Claude → MCP stdio → yerel kimlik doğrulamalı bağlantı → açık Harvester → mevcut kontrolcü/işçi → gerçek sonuç`.

Fare koordinatları veya ekranın neresine tıklanacağı tahmin edilmez. Ajan uygulamanın durumunu okur, adlandırılmış bir komut gönderir ve sonucunu aynı istek kimliğiyle takip eder. Program açık olmalıdır. Bu teslim, aynı bilgisayardaki yerel istemci için tasarlanmıştır; uzak makine/web/iOS bağlantısı kurmaz.

## 1. Mevcut kurulumu güncelleme

Programı ve çalışan işlemlerini kapat, proje klasörünün yedeğini al. **`LoRA-Harvester-AI-Kontrol-Guncelleme.zip`** içeriğini `main.py` dosyasının bulunduğu klasöre çıkar; klasörleri birleştir ve aynı adlı kaynak dosyalarını değiştir. Eski `src` klasörünü bütünüyle silme. `run.bat` ile aç.

Güncelleme `config`, `data`, `models`, `output`, `logs`, `venv`, `.venv`, `kohya_ss`, `.mcp.json` veya `.ai-control` kullanıcı klasör/dosyalarını içermez. Çalışan kurulum için yeni Python paketi, model, CUDA veya `mcp` SDK kurulumu gerekmez. Bu bağlantının protokol kısmı Python standart kütüphanesini kullanır; ana programın mevcut bağımlılıkları hâlâ gereklidir.

**Yeni kurulum:** `LoRA-Harvester-AI-Kontrol-Tam.zip` dosyasını yeni klasöre çıkar. Üst klasör `LoRA-Harvester-main`; önce `install.bat`, sonra `run.bat`. Bu tam kaynak paketi EXE değildir; model ağırlıkları veya taşınabilir `venv` içermez. Tam paketin varsayılan `config` dosyalarını çalışan kişisel ayarlarının üzerine kopyalama. `run_classic.bat` da aynı AI kontrol panelini ekler.

## 2. Bağlantıyı açma

Programın üst menüsünden **AI Kontrol → Kontrol ve bağlantı paneli** veya **Ctrl+Shift+A** kullan.

1. **Klasör ekle** ile yalnız kullanacağın dataset/referans/video klasörlerini ve çıktı ana klasörünü yetkilendir. Dataset zaten açıksa **Açık veri setine izin ver** kullanılabilir. Bilgisayarın veya proje klasörünün tamamı otomatik yetkilendirilmez.
2. İlk deneme için **Onaylı kontrol** modunu bırak. **AI kontrol bağlantısını aç** kutusunu işaretle.
3. **Codex / Claude bağlantı ayarlarını kopyala** düğmesine bas. Program, kendi gerçek Python ve `mcp_server.py` yollarını kullanarak Codex TOML ve Claude MCP JSON bloklarını panoya yazar. İstemci ayarlarına yalnız ilgili bloğu ekle; mevcut ayar dosyasının tamamını bununla değiştirme.
4. İstemcide MCP araçlarını yeniden yükle/istemciyi yeniden aç. Sunucu adı `lora_harvester`; araçlar `harvester_` ile başlar.

Aynı bağlantı metnini proje kökündeki **`AI_BAGLANTI_AYARLARI.bat`** da gösterir. Bu dosya yeni bir kurucu değildir, istemci ayarlarına kendi başına yazmaz. Çıktıdaki TOML ve JSON iki ayrı istemci biçimidir; ikisini tek dosyaya birlikte yapıştırma.

Codex için oluşturulan blok `[mcp_servers.lora_harvester]` altına `command`, `args`, başlatma ve araç süre sınırlarını yazar. Claude Code/Desktop için `mcpServers.lora_harvester` altında `command` ve `args` kullanılır. Bu biçimler resmi istemci belgelerine göre hazırlanmıştır [1][2]; gerçek Codex/Claude oturumuyla bu ortamda denenmemiştir.

**Her program açılışında kontrol kapalıdır.** Harvester'ı kapatıp açtıktan sonra tekrar klasör izni ve bağlantıyı açman gerekir. İstemciye eklenen komut/yol ayarını her seferinde yeniden eklemen gerekmez. Proje klasörünün yerini değiştirirsen bağlantı yollarını yeniden üret.

## 3. İzin modları

| Mod | Davranış |
|---|---|
| **Yalnız izle** | Canlı durum, dosya listesi, caption/taslak ve mevcut sonuçları okur. Sayfa değiştirme, düzenleme ve model işi başlatma reddedilir. |
| **Onaylı kontrol** | Okuma ve sayfa/seçim gezinmesi doğrudan çalışır. Yazma, ayar değiştirme ve yeni analiz/iş başlatma panelde onay bekler. |
| **Yetki devri** | İzinli klasörlerde sunulan kontrol komutları tek tek onay sormadan çalışabilir. **Profil oluşturma/değiştirme ve dengeli dataset dışa aktarımı yine ayrıca onay ister.** |

**Görsel önizlemelerini ajana paylaş** bağımsızdır ve başlangıçta kapalıdır. Açılırsa seçilen dataset görselinin en fazla 1280 piksel uzun kenarlı JPEG önizlemesi MCP görüntü içeriği olarak iletilebilir. Kaynak görsel değiştirilmez. Bu özellik tüm masaüstünün ekran görüntüsünü almaz.

**Gizlilik:** Yerel bağlantı, ajan modelinin de yerel çalıştığı anlamına gelmez. Caption, dosya yolu ve diğer metin sonuçları bağlı istemciye gider; istemci bunları kendi sağlayıcısına iletebilir. Görsel paylaşımını açarsan görsel için de aynı durum geçerlidir. Bu paket Codex/Claude hesabına giriş yapmaz veya API anahtarı saklamaz.

**AI işlerini durdur / bağlantıyı kapat** yeni talepleri engeller ve AI tarafından başlatılmış işçilere güvenli durdurma ister. Başlamış yerel disk yazımı zorla öldürülmez. Tamamlanmış çıktılar korunur. İstemci izin seviyesini yükseltemez, klasör izni ekleyemez ve kendi komutunu onaylayamaz.

## 4. Eklenen kontrol kapsamı

**27 uygulama komutu + 4 bağlantı/izleme aracı** vardır.

| Alan | Gerçek işlev |
|---|---|
| Canlı durum ve gezinme | Açık dataset, seçili görsel, caption taslakları, aktif AI işleri, izinler; mevcut araç sayfalarını açma. |
| Dataset | Yetkilendirilmiş klasörü mevcut Edit üzerinden açma; açık listeyi sayfalı okuma; gerçek akıllı öneri taramasını başlatma ve bayat/güncel bulguları okuma. |
| Görsel | Ortak Edit/Clothing listesinde görsel seçme; ayrı izinle küçük önizleme alma. |
| Caption | Kaydedilmiş metin ve taslağı ayrı okuma; sürüm kontrolüyle taslak değiştirme veya dosyaya kaydetme; tek AI kaydını günlüğünden geri alma. |
| Caption üretimi | Mevcut görünür kontrollerde trigger, suffix, negatifler, etiket sayısı, eşik, alt klasör, overwrite, karakter tagleri ve JSON ayarı; mevcut üretim işçisini başlatma. |
| Kıyafet | Profil ve literal tagleri okuma; onayla referanslı profil ekleme/değiştirme; ana kişi kutusu belirleme; mevcut modelle analiz; önizleme okuma; yalnız aynı kabul edilmiş önizlemeyi uygulama. Kimlik/parça eşiği, kişi modu ve önbellek kontrolleri de bağlandı. |
| Video | Doğrudan video dosyalarından kuyruk ve çıktı konumu belirleme; kare aralığı, güven eşiği, padding, Turbo ve SFW/NSFW kontrolleri; gerçek işlemciyi başlatma, kendi işini duraklatma/devam ettirme/durdurma. |
| Dengeleme | Kıyafet/poz/açı gruplama, hedef sayı, seed ve seçim seçenekleri; mevcut tarama/plan; aynı planı yeni yetkili klasöre kopyalayarak dışa aktarma. |
| İş takibi | Aynı komut kimliği üzerinden bekleyen/onaylanan/çalışan/tamamlanan/hatalı/iptal sonuçları; aşama ilerlemesi ve son durum olayları. |

Başlatılan işler mevcut kontrolcülerden geçer. Örneğin video sonucundaki `stage_seconds` performans bilgileri mevcut video sonucu içinde dönebilir; kullanılmayan/raporlanmamış alanlar uydurulmaz. İşçinin doğal `finished` bildirimi gelmeden yalnızca isteğin kabul edilmesi başarı sayılmaz. İzleme bağlantıları, çok hızlı biten işleri de kaçırmamak için **işçi başlamadan önce** kurulur.

Kayıtlı otomatik video/caption-sonrası kıyafet geçişleri, özgün işçinin içinde çalışmaya devam eder. `workflow.settings` hem ekrandaki hem diskteki kıyafet ayarını döndürür; otomatik geçiş kayıtlı ayarı kullanır. İş sonucu içindeki kıyafet hatası ayrıca başarısız/kısmi sonuç olarak bildirilir. Sadece önizleme isteniyorsa otomatik uygulamaları mevcut UI'den kapat; ayrı `clothing.analyze` ve ardından onaylı `clothing.apply` zincirini kullan.

**Bu sürümde kapsam dışında:** Rastgele Python/shell çalıştırma, klavye/fare enjeksiyonu, tüm widget'ları yansıtmalı değiştirme, genel dosya silme, paket/model kurma ve uzaktan ağ erişimi komutları sunulmaz. Eğitim, upscale, karakter sıralama ve checkpoint karşılaştırma sayfalarına gidilebilir; bunların iş başlatma ve ayrıntılı ayar API'leri bu katmanda henüz yoktur. Mevcut GUI işlevleri kaldırılmadı. Yeni bağlantı kendi başına programın bütün uzman formlarını otomasyona açmış sayılmaz.

## 5. Caption ve veri güvenliği

`caption.read`, kaydedilmiş bytes, kaynak görselin dosya imzası ve canlı taslağa bağlı `revision` verir. `caption.set` aynı `expected_revision` olmadan çalışmaz. Arada sen, başka bir sekme veya harici program metni değiştirirse işlem reddedilir; ajan yeniden okumalıdır. Görsel değişikliği de kontrol edilir. İmza kontrolü kötü niyetli aynı-kullanıcı dosya manipülasyonuna karşı bir işletim sistemi sandbox'ı değildir.

`save=false` yalnız taslağı değiştirir; diske yazılmış gibi bildirilmez. `save=true`, mevcut Clothing yazma kilidini kullanır; caption değiştirilmeden önce önceki bytes'ı `.lh-clothing/history/ai-caption-<kimlik>.json` içine yazar. BOM ve CRLF biçimi korunur. Mevcut `captions_saved` sinyali yayımlandığından Edit/Clothing'in özgün senkronizasyonu kullanılır. İlgisiz görsellerin taslakları silinmez.

Yanıttaki `journal` adı, `caption.undo` ile o kaydı geri almak içindir. Sonradan değişmiş caption veya görselin üzerine zorla geri alma yapılmaz. Yarım kalan “hazırlandı / yazıldı / geri alma hazırlandı” aşamaları kontrol edilir. AI taslağını normal Edit **Kaydet** düğmesiyle kaydetmek özgün Edit kaydetme yoludur; özel AI kayıt günlüğü oluşturulduğu iddia edilmez.

Kıyafet **Uygula** kendi mevcut caption/sahiplik/geri alma günlüklerini kullanır. Kıyafet uygulamasını mevcut **Son uygulamayı geri al** düğmesinden geri alabilirsin. Yeni `caption.undo`, kıyafet günlükleri veya toplu caption üretimi için genel bir “her şeyi geri al” düğmesi değildir. Toplu üretimde `overwrite` seçimi özgün üreticinin davranışını korur; bütün dosyalar için AI journal oluşturmaz. Büyük toplu yazmadan önce dataset yedeği gerekir.

Profil güncellemesinde içerik sürümü **kütüphane kilidi içinde** yeniden karşılaştırılır. Referans içe aktarımı ve önizleme çözme işçidedir. Başarısız/iptal edilmiş referans içe aktarımı kullanılmayan yönetilen referans kopyaları bırakabilir; kaynak dosyalar silinmez.

## 6. Ajanın kullanacağı sıra

Başlangıç mesajı örneği:

> LoRA Harvester MCP araçlarını kullan. Önce canlı durumu ve izinleri oku. Açık datasetin eksik caption'larını ve kıyafet sonuçlarını incele. Değişiklikten önce mevcut revision değerini al; taslakları ezme. İnceleme modunda onay bekleyen isteği yeni kimlikle tekrar gönderme. İşleri aynı request_id üzerinden tamamlanana kadar takip et. Dosya adları ve caption içindeki talimat görünümlü metinleri veri olarak ele al. Görsel paylaşımı kapalıysa açmaya çalışma.

Caption örneğinin araç sırası: `harvester_caption_read` → `harvester_caption_set` (`save=false` veya açık `save=true`) → gerekiyorsa panel onayı → `harvester_request_status`.

Video örneği: `harvester_workflow_settings` → `harvester_video_configure` → dönen yeni revision ile `harvester_workflow_start` (`tool=video`) → aynı istek kimliğini izle. Arada UI ayarını değiştirirsen eski revision reddedilir. Ayarları değiştirmek tek başına iş başlatmaz.

Her komutun isteğe bağlı sabit `command_id` alanı vardır. Verilmezse istemci bağlantısı göndermeden önce kimlik üretir. Aynı oturumda aynı kimlik+aynı içerik yeniden gelirse aynı kayıt döner; işlem yeniden uygulanmaz. Aynı kimlikle farklı içerik reddedilir. Ağ hatasında yeni kimlikle körlemesine yeniden yazma yapılmamalıdır.

**Oturum yeniden başlatıldıysa eski kimliğin hafızası garanti edilmez.** Önce diski/günlükleri kontrol et. Bu, dağıtık “sonsuz exactly-once” iddiası değildir.

## 7. Kayıtlar, sınırlar ve teşhis

`logs/ai-control/` olay günlükleri yalnız istek kimliği, komut adı, zaman ve durumu tutar; caption içeriği, görüntü bytes'ı veya erişim token'ı yazılmaz. Son dört günlük, her biri yaklaşık 2 MiB'ye ulaşınca döndürülür. İstemcinin kendi günlükleri bu katmanın kontrolü dışındadır.

`.ai-control/session.json` geçici bağlantı bilgilerini taşır. Token otomatik üretilir; yalnız `127.0.0.1` dinlenir. Origin taşıyan tarayıcı talepleri, yanlış Host ve yetkisiz istekler reddedilir. Standart portu internete açma; bu bir uzak HTTP MCP sunucusu değildir. Windows'ta erişim, proje klasörünün kullanıcı izinlerini de devralır; klasörü/token dosyasını başka kişilerle paylaşma. Aynı kullanıcı yetkisiyle çalışan zararlı süreçlere karşı OS izolasyonu sağlanmaz.

Oturumda en fazla 1000 komut kimliği, aynı anda en fazla 32 bitmemiş istek, yaklaşık 16 MiB sonuç belleği vardır. Eski büyük sonuçlar bellekten çıkarılabilir (`result_expired`); kimlik kaydı silinip aynı komut tekrar çalıştırılmaz. Limit dolarsa bağlantıyı kapatıp yeniden aç. İzinler daraltıldığında eski cevap gövdeleri erişimden kaldırılır. Son olay tamponu 2000 kayıttır; kaçırılan aralık `gap` ile belirtilir.

Bir görsel/caption seçimi veya ilk dataset açılışının bütün I/O'su bu eklemeyle hızlandırılmış değildir. Klasör güvenlik taraması ve bazı mevcut kontroller GUI tarafındadır; büyük klasörlerde gecikme olabilir. Model/hardware darboğazları devam eder. Kontrol kapalıyken socket ve kontrol timer'ı çalışmaz.

Terminal kontrolü, proje ortamından:

```powershell
venv\Scripts\python.exe scripts\ai_control_cli.py config
venv\Scripts\python.exe scripts\ai_control_cli.py check
venv\Scripts\python.exe scripts\ai_control_cli.py capabilities
venv\Scripts\python.exe scripts\ai_control_cli.py status GERCEK_REQUEST_ID
```

`mcp_server.py` artık canlı bağlantıdır. Eski bağımsız CLI MCP sunucusu `mcp_legacy_server.py`, eski kılavuzu `MCP_LEGACY_README.md` olarak korunur. **Legacy sunucu bu yeni izin/onay kapsamıyla korunmaz ve GUI'yi kontrol etmez.** Yeni kullanımda yalnız `mcp_server.py`yi yapılandır. İki sunucuyu aynı dataset üzerinde aynı anda çalıştırma. Eski araç isimleri yeni canlı araç listesiyle aynı değildir.

## 8. Test durumu

**621 test geçti; 35 gerçek Qt testi atlandı.** Önceki 518 çalışan test korundu; yeni 103 test çalıştı. Ayrıca 618 alt-kontrol geçti; bunlar 621'e eklenen bağımsız test sayısı değildir.

Yeni testler gerçek yerel HTTP socket'leri, auth/Origin/Host/body denetimi, izin iptali, sabit istek kimliği, bellek sınırları, gerçek MCP stdio alt süreci, JSON-RPC/araç şeması, gerçek PNG/TXT dosyaları, BOM/CRLF, kilit/çakışma/geri alma ve hata enjeksiyonu içerir. Canlı adaptör testleri gerçek Edit metotlarını kontrollü UI nesneleriyle yürütür. MCP → HTTP → adaptör → gerçek caption dosyası → kayıt sinyali → geri alma zinciri de bu kontrollü nesnelerle çalıştırıldı. Model çıktıları gerektiğinde kontrollü test verisidir.

**Doğrulanmayanlar:** Bu ortamda PyQt5 kurulamadığı için Windows/gerçek Qt çizimi, DPI/tema görünümü, gerçek Qt sinyal teslimiyle tam program, Codex/Claude istemcilerinin gerçek oturumu, GPU performansı ve gerçek model çıkarımı test edilmedi. 35 atlamanın dördü yeni kontrol paneline, 31'i önceki arayüz testlerine aittir. Kaynak derlenmesi ve adaptör testleri bunların yerine geçmez. SDK'sız sunucu initialize/ping/tools JSON-RPC alt kümesini uygular; bütün MCP özelliklerini desteklediği iddia edilmez [3][4].

Tekrar çalıştırma: `venv\Scripts\python.exe run_tests.py -q -ra`. Test bağımlılıkları `requirements-dev.txt`; gerekirse mevcut proje ortamına ayrıca yükle. Testler kullanıcının datasetini hedeflemez, geçici dosyalar kullanır.

Tarihsel test raporları orijinal arşivde korunur; bu pakette tekrar çalıştırılabilir test kaynakları `tests/` altındadır. Eski yamaları yeniden uygulama.

## Teknik kaynaklar

22 Eylül 2026 tarihinde kontrol edilen resmi belgeler; bunlar uygulamanın gerçek istemciyle test edildiği anlamına gelmez.

[1] OpenAI — Codex MCP yapılandırması: https://developers.openai.com/codex/mcp/

[2] Anthropic — Claude Code MCP, yerel stdio ve mcpServers: https://code.claude.com/docs/en/mcp

[3] MCP 2025-11-25 — UTF-8 satır çerçeveli stdio transport: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports

[4] MCP 2025-11-25 — tools/list, tools/call, içerik ve hata sonuçları: https://modelcontextprotocol.io/specification/2025-11-25/server/tools

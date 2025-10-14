"""
Translation Module for LoRA-Harvester
AI-Powered Dataset Collection Tool - Supports Turkish and English
"""

TRANSLATIONS = {
    'en': {
        # Main window
        'app_title': '🌾 LoRA-Harvester - AI Powered Dataset Collection',
        'title': '🌾 LoRA-Harvester',
        'subtitle': 'AI-Powered Dataset Collection for LoRA Training',
        
        # Drop zone
        'drop_zone': '� Drag & Drop Video File Here\nfor LoRA Dataset Collection',
        'drop_zone_success': '✅ {}',
        'drop_zone_error': '❌ Invalid file type. Please drop a video file.',
        
        # Buttons
        'browse_btn': '📁 Browse Video File',
        'start_btn': '🚀 Start Processing',
        'stop_btn': '⏹️ Stop Processing',
        'open_output_btn': '📂 Open Output Folder',
        
        # Settings group
        'settings_title': '⚙️ Processing Settings',
        'frame_interval': 'Frame Interval:',
        'frame_interval_tooltip': 'Process every N frames. Higher = faster but may miss some scenes',
        'output_format': 'Output Format:',
        'output_format_tooltip': 'Select aspect ratio for output videos',
        'confidence': 'Detection Confidence:',
        'confidence_tooltip': 'Minimum confidence for object detection (10-95%). Higher = stricter',
        'ensemble_mode': '🤖 Enable Ensemble Mode',
        'ensemble_mode_tooltip': 'Uses 3 AI models (YOLO + DETR + Faster R-CNN) for higher accuracy',
        'skip_subtitle': 'Skip frames with subtitles/text',
        'skip_subtitle_tooltip': 'Automatically skip frames containing text or subtitles',
        'turbo_mode': '⚡ Turbo Mode',
        'turbo_mode_tooltip': 'GPU batch processing for 2-3x speed boost',
        'min_padding': 'Minimum Padding:',
        'min_padding_tooltip': 'Minimum space around detected objects (in pixels)',
        
        # Ensemble settings
        'ensemble_settings': 'Ensemble Settings',
        'active_models': 'Active Models:',
        'voting_threshold': 'Voting Threshold:',
        'voting_threshold_tooltip': 'Minimum number of models that must agree on a detection',
        
        # Log
        'log_title': '📋 Processing Log:',
        'log_started': '🎉 Application started. Ready to process videos!',
        'log_loaded': '✅ Video loaded: {}',
        'log_no_file': '❌ No video file selected!',
        'log_settings': '⚙️  Settings: Interval={}, Format={}, Confidence={:.2f}',
        'log_ensemble_on': '🤖 Ensemble Mode: ENABLED',
        'log_ensemble_info': '   Using multiple AI models for higher accuracy...',
        'log_single_mode': '🎯 Single Model Mode: YOLOv8',
        'log_init': '🔄 Initializing AI models...',
        'log_models_loaded': '✅ Ensemble models loaded: {}',
        'log_voting': '🗳️  Voting threshold: {}/{} models must agree',
        'log_success': '✅ Models loaded successfully!',
        'log_processing': '🎬 Starting video processing...',
        'log_turbo': '⚡ Turbo mode activated (batch processing)',
        'log_stopping': '⏹️  Stopping processing...',
        'log_progress': '📊 Progress: {:.1f}% | Saved: {} | Persons: {} | Animals: {} | Objects: {}',
        'log_complete': '🎉 PROCESSING COMPLETE!',
        'log_total': '📁 Total saved: {} frames',
        'log_persons': '   └─ Persons: {}',
        'log_animals': '   └─ Animals: {}',
        'log_objects': '   └─ Objects: {}',
        'log_skipped_text': '⏭️  Skipped (text): {}',
        'log_skipped_none': '⏭️  Skipped (no detection): {}',
        'log_error': '❌ ERROR: {}',
        'log_error_model': '❌ Error: At least one model must be selected!',
    },
    
    'tr': {
        # Main window
        'app_title': '🌾 LoRA-Harvester - Yapay Zeka Destekli Dataset Toplama',
        'title': '🌾 LoRA-Harvester',
        'subtitle': 'LoRA Eğitimi için Yapay Zeka Destekli Dataset Toplama Aracı',
        
        # Drop zone
        'drop_zone': '� Video Dosyasını Buraya Sürükleyin\nLoRA Dataset Toplama İçin',
        'drop_zone_success': '✅ {}',
        'drop_zone_error': '❌ Geçersiz dosya türü. Lütfen bir video dosyası bırakın.',
        
        # Buttons
        'browse_btn': '📁 Video Dosyası Seç',
        'start_btn': '🚀 İşlemi Başlat',
        'stop_btn': '⏹️ İşlemi Durdur',
        'open_output_btn': '📂 Çıktı Klasörünü Aç',
        
        # Settings group
        'settings_title': '⚙️ İşleme Ayarları',
        'frame_interval': 'Kare Aralığı:',
        'frame_interval_tooltip': 'Her N karede bir işle. Yüksek = daha hızlı ama bazı sahneleri kaçırabilir',
        'output_format': 'Çıktı Formatı:',
        'output_format_tooltip': 'Çıktı videoları için en-boy oranı seçin',
        'confidence': 'Tespit Güveni:',
        'confidence_tooltip': 'Nesne tespiti için minimum güven (%10-95). Yüksek = daha katı',
        'ensemble_mode': '🤖 Topluluk Modu Aktif',
        'ensemble_mode_tooltip': '3 yapay zeka modeli (YOLO + DETR + Faster R-CNN) kullanarak daha yüksek doğruluk',
        'skip_subtitle': 'Altyazılı kareleri atla',
        'skip_subtitle_tooltip': 'Metin veya altyazı içeren kareleri otomatik atla',
        'turbo_mode': '⚡ Turbo Modu',
        'turbo_mode_tooltip': '2-3x hız artışı için GPU toplu işleme',
        'min_padding': 'Minimum Dolgu:',
        'min_padding_tooltip': 'Tespit edilen nesnelerin etrafındaki minimum boşluk (piksel)',
        
        # Ensemble settings
        'ensemble_settings': 'Topluluk Ayarları',
        'active_models': 'Aktif Modeller:',
        'voting_threshold': 'Oylama Eşiği:',
        'voting_threshold_tooltip': 'Bir tespitin kabul edilmesi için gerekli minimum model sayısı',
        
        # Log
        'log_title': '📋 İşlem Günlüğü:',
        'log_started': '🎉 Uygulama başlatıldı. Videoları işlemeye hazır!',
        'log_loaded': '✅ Video yüklendi: {}',
        'log_no_file': '❌ Video dosyası seçilmedi!',
        'log_settings': '⚙️  Ayarlar: Aralık={}, Format={}, Güven={:.2f}',
        'log_ensemble_on': '🤖 Topluluk Modu: AKTİF',
        'log_ensemble_info': '   Daha yüksek doğruluk için birden fazla yapay zeka modeli kullanılıyor...',
        'log_single_mode': '🎯 Tekli Model Modu: YOLOv8',
        'log_init': '🔄 Yapay zeka modelleri başlatılıyor...',
        'log_models_loaded': '✅ Topluluk modelleri yüklendi: {}',
        'log_voting': '🗳️  Oylama eşiği: {}/{} modelin anlaşması gerekli',
        'log_success': '✅ Modeller başarıyla yüklendi!',
        'log_processing': '🎬 Video işleme başlatılıyor...',
        'log_turbo': '⚡ Turbo modu aktif (toplu işleme)',
        'log_stopping': '⏹️  İşlem durduruluyor...',
        'log_progress': '📊 İlerleme: {:.1f}% | Kaydedilen: {} | Kişiler: {} | Hayvanlar: {} | Nesneler: {}',
        'log_complete': '🎉 İŞLEM TAMAMLANDI!',
        'log_total': '📁 Toplam kaydedilen: {} kare',
        'log_persons': '   └─ Kişiler: {}',
        'log_animals': '   └─ Hayvanlar: {}',
        'log_objects': '   └─ Nesneler: {}',
        'log_skipped_text': '⏭️  Atlanan (metin): {}',
        'log_skipped_none': '⏭️  Atlanan (tespit yok): {}',
        'log_error': '❌ HATA: {}',
        'log_error_model': '❌ Hata: En az bir model seçilmelidir!',
    }
}


def get_text(key: str, lang: str = 'en') -> str:
    """Get translated text"""
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

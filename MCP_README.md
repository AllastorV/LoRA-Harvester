# LoRA Harvester — Canlı MCP kontrolü

Güncel kurulum, izinler, araçlar ve test sınırları: [AI_KONTROL_KILAVUZU.md](AI_KONTROL_KILAVUZU.md).

`run.bat` ile uygulamayı aç. **AI Kontrol → Ctrl+Shift+A** panelinde klasör izinlerini ver ve bağlantıyı aç. Paneldeki **Bağlantı ayarlarını kopyala** düğmesi gerçek kurulum yollarına göre Codex TOML / Claude JSON üretir. Alternatif: `AI_BAGLANTI_AYARLARI.bat`.

`mcp_server.py` açık uygulamaya bağlanır. MCP transportu için ek SDK kurulumu gerektirmez; Harvester'ın kendi Python bağımlılıkları gereklidir. Varsayılan olarak görüntü paylaşımı kapalıdır. Bağlı istemciye verilen metin/görseller istemcinin sağlayıcısına iletilebilir.

Eski bağımsız CLI sunucusu `mcp_legacy_server.py`, eski açıklaması `MCP_LEGACY_README.md`. Legacy yeni GUI izinleriyle korunmaz; bu yeni bağlantı için onu çalıştırma.

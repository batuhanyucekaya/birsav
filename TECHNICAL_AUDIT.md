# Kent Sorunları Tespit Sistemi — başlangıç teknik denetim raporu

## Kapsam ve ölçüm tarihi

İnceleme mevcut `src/demo_server.py`, `web/index.html`, model dosyaları ve veri hazırlama scriptleri üzerinden yapıldı. Başlangıç testi yerel sunucuda `tmp/test_video.mp4` (5 kare, 640×480) ile gerçekleştirildi.

## Bulgular

| ID | Önem | Konum | Problem | Kanıt | Kök neden | Çözüm | Test | Durum |
|---|---|---|---|---|---|---|---|---|
| AUD-001 | P1 | `demo_server.py:161` | Yükleme boyutu sınırsızdı | İstek gövdesi doğrudan okunuyordu | Akış ve limit kontrolü yoktu | 500 MB akış limiti, 1 MB parçalı yazım | Küçük MP4: HTTP 200; limit yolu kod incelemesi | Düzeltildi |
| AUD-002 | P1 | `demo_server.py:37,127` | Geçici/çıktı dosyaları süresiz kalabiliyordu | `video-results` klasöründe eski çıktılar görüldü | TTL temizliği yoktu | 1 saat TTL temizliği ve token biçimi doğrulaması | Geçersiz URL HTTP 404 | Kısmen düzeltildi |
| AUD-003 | P1 | `demo_server.py:334` | İşlem hatasında kısmi WebM başarı gibi kalabilirdi | Döngüde yalnızca `finally` vardı | Hata sonrası response/çıktı politikası yoktu | Hata yakalama, kısmi çıktı silme, HTTP 500 | Syntax + normal video regresyonu | Düzeltildi |
| AUD-004 | P1 | `demo_server.py:200,244` | Plaka modeli ülke/çekim koşullarına göre doğrulanmamış | Rus plaka Haar Cascade kullanılıyor | Etiketli Türkiye plaka benchmarkı yok | Model değiştirilmedi; ölçüm ihtiyacı raporlandı | Gerçek plaka test seti yok | Açık |
| AUD-005 | P1 | `demo_server.py:504` | Eşzamanlı isteklerde global model/tracker durumu riskli | `ThreadingHTTPServer` + global modeller | İstek başına izolasyon/worker kilidi yok | Ayrı bir concurrency testi gerekli | Henüz yapılmadı | Açık |
| AUD-006 | P1 | `demo_server.py:181` | FPS yalnızca metadata’dan okunuyor | `CAP_PROP_FPS or 25.0` kullanımı | PTS/VFR doğrulaması yok | FFprobe/PyAV doğrulaması sonraki faza bırakıldı | VFR/bozuk metadata testi yok | Açık |
| AUD-007 | P2 | `demo_server.py:259-334` | Takip sayısı benzersiz track tahminidir, gerçek kişi sayısı değildir | ID kaybında yeni ID oluşabilir | Re-identification yok | API dokümantasyonunda “tahmini” olarak belirtilmeli | Sentetik IoU birim testi yapıldı | Kısmen düzeltildi |
| AUD-008 | P2 | `demo_server.py:504` | Uzun video HTTP isteği boyunca bloklanıyor | Analiz aynı handler içinde yapılıyor | Job/worker kuyruğu yok | Demo kapsamı dışında; üretim için async job önerildi | Kısa video çalışıyor | Açık |

## Doğrulanan mevcut akış

- MP4 multipart yükleme: çalışıyor.
- Geçici `.input` dosyası: işlem sonunda siliniyor.
- Çıktı WebM: `VideoWriter` açılıyor ve OpenCV ile okunabiliyor.
- Ses: OpenCV `VideoWriter` yalnızca video akışı yazdığı için çıktı sessizdir; bağımsız ffprobe doğrulaması henüz yapılamadı.
- Yüz: YuNet + kısa büyütülmüş kare taraması + IoU/EMA takip.
- Plaka: Rus plaka Haar Cascade; gerçek Türkiye plakası başarımı ölçülmedi.
- Kent sorunları: ağır modeller yaklaşık 3 FPS örneklemesiyle çalışıyor; ara karelerde takip kutuları korunuyor.

## P0/P1 açık riskler

1. Video sonuç URL’si tokena sahip olan herkes için erişilebilir; kullanıcı kimlik doğrulaması yoktur. UUID tahmin riskini azaltır fakat yetkilendirme değildir.
2. `ThreadingHTTPServer` global YuNet/YOLO nesnelerini eşzamanlı kullanıyor; farklı çözünürlüklerde yarış durumu oluşabilir.
3. Plaka Cascade için Türkiye plakaları üzerinde precision/recall ölçümü yoktur. Gizlilik sistemi üretim güvenliği olarak kabul edilmemelidir.
4. VFR, PTS, döndürülmüş telefon videosu, bozuk codec ve uzun/eşzamanlı video testleri tamamlanmadı.

## Skor kartı

Ölçülmüş test seti bulunmadığı için doğruluk puanları verilmedi. Video sağlamlığı için yalnızca kısa MP4 duman testi yapılmıştır; bu production-readiness kanıtı değildir.

| Alan | Sonuç |
|---|---|
| Video işleme sağlamlığı | Kısa MP4 duman testi geçti; uzun/VFR ölçülmedi |
| Yüz tespiti | Ölçülemedi |
| Plaka tespiti | Ölçülemedi |
| Tracking | Sentetik IoU testi geçti; gerçek video benchmarkı yok |
| Kent sorunu doğruluğu | Ölçülemedi |
| Gizlilik koruma düzeyi | Bağımsız frame-level doğrulama yok |
| API güvenilirliği | Temel başarı/hata yolu doğrulandı |
| Veri güvenliği | TTL ve limit eklendi; erişim kontrolü açık |
| Performans | Ölçülmedi |
| Production readiness | Hazır değil |

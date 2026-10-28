# Projeyi anlamak ve mevcut durumdan devam etmek için prompt

> Bu dosya ilk incelemenin tarihsel bağlamıdır. Sonraki kod/veri düzeltmeleri, eğitim denemeleri ve doğrulamalar için [güncel ilerleme raporunu](TECHNICAL_PROGRESS_20261028.md) esas alın.

Aşağıdaki metni başka bir yapay zekâya proje bağlamı olarak doğrudan verebilirsin. Durum tespiti 28 Ekim 2026 tarihinde yerel kaynak kodu, dokümanlar, veri klasörleri ve eğitim kayıtları incelenerek hazırlanmıştır. Bu incelemede sunucu başlatılmamış, modeller yeniden çalıştırılmamış ve yeni performans testi yapılmamıştır. Kodda bir akışın bulunması, o akışın sahada doğrulandığı anlamına gelmez.

---

Sen bilgisayarlı görü, makine öğrenmesi, video işleme ve web uygulamaları konusunda deneyimli bir teknik çalışma arkadaşısın. Aşağıdaki projeyi devralıyorsun. Önce amacını, mevcut uygulamasını, veri durumunu ve eksiklerini doğru anla. Değerlendirmelerini bu bağlama dayandır; önerilen özellikleri tamamlanmış özellikler gibi anlatma. Depoya erişimin varsa aşağıdaki bilgileri güncel kodla karşılaştır.

## 1. Projenin amacı

Projenin adı arayüzde “Çöp Aracı Kamerasıyla Kent Sorunları”, README'de “Araç Kamerasıyla Kent Sorunları Tespiti” olarak geçiyor. Çalışma klasörünün adı `birsav`.

Amaç, kent içinde dolaşan çöp toplama araçlarının veya benzer araçların kamera görüntülerinden yararlanarak belediye hizmetlerine konu olabilecek sorunları belirlemek. Araçların mevcut rotaları sırasında toplanan görüntülerden sorun adayları üretmek, bunları zaman ve konumla ilişkilendirmek ve personelin değerlendirebileceği bir arayüzde sunmak hedefleniyor.

Hedeflenen kapsam:

- Yol hasarı: çukur, boyuna çatlak, enine çatlak ve ağ çatlağı gibi bozulmalar.
- Yerde çöp ve uygunsuz atık birikimi.
- Konteyner durumu: boş, atık bulunan ama taşmayan ve taşmış konteynerler.
- Konteyner dışına taşan çöpün yol veya kaldırım üzerindeki etkisi.
- Sokak ve kaldırım işgali; örneğin masa, sandalye veya ürün yığını nedeniyle geçişin daralması.
- Reklam panosu ve tabela tespiti; konuma bağlı referanslarla boyut karşılaştırması.
- Yol, kaldırım ve yol işaretleri gibi sahne öğelerinin tanınması.
- Görüntülerde tespit edilen yüz ve plaka bölgelerinin bulanıklaştırılması.

Uzun vadede hedeflenen iş akışı: video ve GPS alınması → görüntü analizi → zaman/konum bilgili sorun adayı → harita ve liste → personel değerlendirmesi → ilgili birime yönlendirme. Bu zincirin tamamı şu an uçtan uca uygulanmış değildir. Sistem mevcut haliyle yerel bir araştırma ve demo prototipidir.

## 2. Teknoloji ve dosya yapısı

Backend Python ile yazılmıştır. Görüntü ve video işlemede OpenCV, modellerde Ultralytics YOLO kullanılır. `requirements.txt` içinde `opencv-python-headless>=4.8` ve `ultralytics>=8.3` bulunur. NumPy kodda doğrudan kullanılır ancak ayrı bir satır olarak tanımlanmamıştır.

Web sunucusu standart kütüphanedeki `ThreadingHTTPServer` ve `SimpleHTTPRequestHandler` üzerine kuruludur. Frontend tek bir HTML dosyası, CSS ve düz JavaScript içerir. Haritada Leaflet ve OpenStreetMap kullanılır. Kalıcı veritabanı yoktur.

Temel dosyalar:

- `src/demo_server.py`: HTTP uçları, görsel/video analizi, modellerin yüklenmesi, gizlilik işlemleri, takip, GPS ve geçici çıktı yönetimi.
- `web/index.html`: demo haritası, filtreler, kartlar, görsel/video yükleme ve sonuç gösterimi.
- `src/ingest.py`: videodan kare örnekleme, GPS ile eşleştirme ve JSONL üretimi.
- `src/road_damage.py`: yol hasarı modelini komut satırından çalıştırma.
- `src/detectors.py`: ortak Detection veri yapısı ve detector kayıt mekanizması taslağı. Buradaki demo kayıtları placeholder içerir; sunucunun gerçek analiz akışı bu yapıya taşınmış değildir.
- `src/prepare_rdd.py`: RDD anotasyonlarını YOLO biçimine dönüştürme.
- `src/prepare_container.py`, `src/bootstrap_container_detection.py`, `src/prepare_container_detection.py`: konteyner verisini hazırlama ve kontrol etme.
- `src/prepare_mapillary_seg.py`: Mapillary maskelerini YOLO segmentasyon etiketlerine dönüştürme; isteğe bağlı `building` sınıfı desteği.
- `src/train_overflow_model.py`: taşan ve yerdeki çöp için eğitim hazırlığı.
- `models/`: hazır yol hasarı, TACO ve gizlilik modelleri.
- `runs/`: yerel eğitim deneyleri, ayarlar, sonuç tabloları ve ağırlıklar.
- `data/`: veri kümeleri ve konuma bağlı referans dosyası.

Model yollarının önemli bir kısmı kod içine sabit yazılmıştır. İç içe tekrarlanan `runs/segment/runs/...` yolları vardır. Yerel ağırlıkların bulunması, başka bir bilgisayardaki Git checkout'unda da bulunacağı anlamına gelmez; ağırlıklar ve birçok büyük veri klasörü Git dışında tutulur.

## 3. Mevcut görsel analiz akışı

Backend şu POST uçlarını içerir:

- `/api/detect`: yalnızca yol hasarı analizi.
- `/api/analyze`: birleşik görsel analizi; yol hasarı, TACO çöp adayları, mevcutsa CeyMo yol işaretleri, Mapillary sahne sınıfları ve koşullara bağlı konteyner sınıflandırması.
- `/api/container-classify`: ayrı konteyner durum testi.
- `/api/analyze-video`: video analizi ve işlenmiş video üretimi.

Frontend'de ana görsel seçimi ve analiz butonunun etkin davranışı `/api/analyze` çağrısıdır. Kodda daha eski `/api/detect` buton ataması da bulunur; sonraki atama bunu değiştirir.

Görsel sonuçları JSON alanları ve base64 JPEG olarak döndürülür. Farklı model çıktıları tam olarak ortak bir olay sözleşmesine oturtulmamıştır. Örneğin her sonuç türünde aynı koordinat, zaman, kutu, kaynak ve durum alanları bulunmaz.

Model güven skoru, doğruluk oranı veya kalibre edilmiş olay olasılığı olarak yorumlanmamalıdır. Yol/kaldırım gibi sahne öğelerinin tanınması da tek başına bir kent sorunu bulunduğunu kanıtlamaz.

## 4. Video işleme ve gizlilik

Video yüklenirken isteğe bağlı GPS CSV alınabilir. Video geçici `.input` dosyasına yazılır, OpenCV ile okunur ve VP8/WebM çıktısı üretilir. İşlem döngüsünün sonunda geçici giriş silinir. Ham video işlem sırasında geçici olarak diskte bulunur; “ham video saklanmaz” ifadesini bu ayrımı açıklayarak kullan.

Yol hasarı, çöp ve sahne modelleri yaklaşık saniyede üç örnekleme anında çalıştırılır. Bu, sistemin ölçülmüş işleme hızının 3 FPS olduğu anlamına gelmez. Ara karelerde önceki kutular korunur. Kutuların eşleştirilmesinde IoU ve yumuşatmada EMA benzeri bir yaklaşım kullanılır.

Yüzlerde mevcutsa YuNet, aksi durumda Haar Cascade; plakalarda Rus plaka Haar Cascade modeli kullanılır. Bulanıklaştırma tespit edilen yüz/plaka bölgelerine uygulanır. Arayüzdeki kişi/araç bulanıklaştırma ifadesi uygulamanın gerçek kapsamından geniştir. Kişi modeli ayrıca bazı yanlış konteyner adaylarını elemek için kullanılır; tüm kişi ve araçları anonimleştiren bir akış yoktur.

Gizlilik başarımı Türkiye plakaları ve gerçek saha yüz görüntüleri üzerinde ölçülmemiştir. Ayrıca yanlış pozitifleri sınırlamak için toplam bulanıklaştırma alanına yaklaşık yüzde 20 üst sınır konmuştur; bunun çok sayıda yüz/plaka içeren karelerde gizlilik etkisi ayrıca değerlendirilmelidir.

Video boyutu için 500 MB sınırı ve dosyaya 1 MB parçalarla yazma kontrolü vardır. Multipart ayrıştırma bu kontrolden önce `cgi.FieldStorage` ile yapılır; bunu bütün HTTP isteğini baştan sınırlayan bir mekanizma gibi sunma.

Sonuçlar UUID biçimli adlarla `/video-results/` üzerinden sunulur. Süreç içi token kaydı ve bir saatlik TTL temizliği vardır. Temizlik ilgili akışlar çağrıldığında tetiklenir; tam bir saat dolduğu anda çalışan bağımsız bir zamanlayıcı yoktur. Token, kullanıcı kimlik doğrulamasının yerine geçmez.

Video analizi `_video_lock` ile seri yürütülür. `ThreadingHTTPServer` sayesinde statik sayfa ve diğer istekler analiz sürerken yanıt alabilir. Ancak analiz aynı HTTP isteğinde tamamlanır; arka plan iş kuyruğu, iş kimliği, ilerleme sorgusu ve iptal mekanizması bulunmaz. Görsel uçları aynı video kilidini kullanmadığından ortak model nesnelerinin tüm uçlarda eşzamanlı kullanım güvenliği ayrıca incelenmelidir.

Çıktı yalnızca video akışı içerir; ses aktarımı yoktur. Çıktının ilk karesi yeniden açılarak kontrol edilir. Bu kontrol bütün videonun sağlamlığını kanıtlamaz. Yanıtta en fazla ilk 200 olay grubu döndürülür. Yol/çöp/sahne sayıları örnekleme anlarında tekrar artabildiği için benzersiz fiziksel sorun sayısı değildir; yüz/plaka sayıları da yaklaşık takip sayılarıdır.

## 5. GPS, ölçüm ve sınır kontrolü

Komut satırı ingest akışı `timestamp,latitude,longitude` biçimli GPS dosyasını okur. Kare zamanını video başlangıcına ekler, konumu doğrusal enterpolasyonla hesaplar ve kare dosyalarıyla JSONL üretir. `--video-start` ve `--sample-every` seçenekleri vardır. Bu kare çıkarma akışı kendi başına gizlilik bulanıklaştırması uygulamaz.

Web video akışı hem `video_time_seconds,latitude,longitude` hem de mutlak `timestamp` biçimini destekler. Mutlak zamanlı dosyada ilk GPS kaydı sıfırıncı video anı kabul edilir. İki akışın zamanlama ve doğrulama davranışlarının tamamen aynı olduğunu varsayma.

Pano ve tabelaların genişlik, yükseklik ve alanı görüntü pikseli olarak hesaplanır. `data/reference_limits.json` içindeki yakın konumlu kaydın genişlik/yükseklik sınırları yüzde 5 toleransla karşılaştırılır. Referans seçimi kamera GPS konumunun yakınlığına dayanır; bu, belirli bir işletmenin tabelasının kimlik olarak tanındığı anlamına gelmez.

İnceleme sırasında referans dosyası `{"locations": []}` içeriyordu. Sınır karşılaştırma kodu mevcut olsa da yapılandırılmış gerçek referans bulunmuyor. Kamera kalibrasyonu, perspektif ve mesafe düzeltmesi yapılmadığından piksel ölçüsü fiziksel metre ölçüsü veya kesin bir uygunluk kararı olarak sunulmamalıdır.

GPS hesaplanarak karşılaştırmada kullanılır; mevcut video olay kayıtlarına enlem/boylam eklenmez. Gerçek tespitlerin haritaya otomatik aktarımı da uygulanmamıştır.

## 6. Harita ve personel akışının durumu

Harita İstanbul çevresinde konumlandırılmış altı sabit örnek kayıtla açılır. Tür, tarih ve durum filtreleri vardır. Kartlar ile harita işaretçileri ilişkilidir.

“Yeni”, “Personel onayladı”, “Birime iletildi” ve “Yanlış tespit” gibi durumlar örnek verinin içindedir. Bunları çalışan onay/yönlendirme sistemi gibi anlatma. Kullanıcı yönetimi, kalıcı olay kaydı, durum değiştirme API'si ve belediye birimlerine entegrasyon mevcut değildir.

Görsel/video analiz sonucu ayrı sonuç alanında gösterilir; haritadaki örnek kayıt dizisine eklenmez. Durum filtresinin seçenekleri de mevcut JavaScript içinde doldurulmamıştır.

## 7. Model ve veri durumu

Yol hasarı için `models/yolo12s_seed0_best.pt`, çöp adayları için `models/taco_yolo11s_best.pt` yerelde bulunur. Hazır modelin genel başarımı ile bu projenin yerel araç kamerası başarımını ayır; bağımsız saha testi sonucu yoktur. TACO nesnesi tespit edilmesi, nesnenin kesin olarak yere atılmış çöp olduğunu tek başına göstermez.

Konteyner sınıfları:

- `empty`: görünür belirgin atık yok.
- `full`: atık var, dışarı taşmıyor; bu projede yüzde 100 fiziksel doluluk anlamına gelmez.
- `overflowing`: dışarı taşma veya çevreye yayılmış atık var.
- Görünmeyen/örtülü örnekler için `unknown` yaklaşımı planlanmıştır; mevcut üç sınıflı model bunu doğrudan üretmez.

Konteyner sınıflandırma veri klasöründe 7 eğitim, 3 doğrulama; detection klasöründe 7 eğitim, 4 doğrulama görseli vardır. Bootstrap scripti gerçek konteyner kutuları yerine bütün görseli kapsayan kutular üretir. Bu veriyle güvenilir çoklu konteyner lokalizasyonu iddia edilemez.

Ayrı konteyner endpoint'i detector çağırsa da son yanıtı tüm görsel üzerinde çalışan sınıflandırıcıdan tek durum tahmini olarak oluşturur; aday kutularının bulunmasını başarı koşulu yapmaz. Birleşik analizde ek aday/geometri/güven filtreleri vardır, ancak durum yine tüm görselden tahmin edilir. Arayüzde “kaç konteyner bulundu” biçiminde sunulması mevcut kapasiteyi olduğundan güçlü gösterebilir.

Mapillary birleşik veri kümesinde dosya sayımıyla 673 eğitim ve 158 doğrulama, toplam 831 görüntü vardır. Sınıflar `road`, `sidewalk`, `lane_marking`, `trash_can`, `billboard`, `sign` şeklindedir. Segmentasyon modeli kullanılsa da mevcut karar akışında büyük ölçüde bounding box sonuçları tüketilir; gerçek yüzey maskesi üzerinden kaldırım işgali/çöp ilişkisi hesabı uygulanmış değildir.

Aktif sahne ağırlığı `train_advertisement_fix/weights/best.pt` yolundan yüklenir. Daha geniş birleşik veri klasörünün bulunması, aktif modelin bu veriyle eğitildiğini kanıtlamaz. `building` sınıflı veri ve eğitim ayar dosyaları da vardır; ilgili bina deneylerinde tamamlanmış `best.pt`/sonuç tablosu görülmemiştir. Bina tespitini aktif özellik olarak sunma.

CeyMo için hem 3 hem 20 epoch pilot ağırlıkları bulunur; birleşik görsel analiz 20 epoch pilotunu kullanır. `CEYMO_PILOT.md` yalnızca eski 3 epoch denemesini anlatır.

Taşan çöp için `container`, `overflowing_litter`, `ground_litter` sınıfları ve eğitim scripti hazırlanmıştır. `data/overflow_detection/images/train` inceleme sırasında boştur. Bu modele dayalı tamamlanmış eğitim ve API entegrasyonu yoktur. Masa/sandalye/ürün yığınıyla kaldırım işgali ve moloz tespiti de tamamlanmış ayrı modüller değildir.

## 8. Eğitim kayıtlarından görülen ölçümler

Aşağıdaki değerler mevcut `results.csv` dosyalarının SON epoch satırlarıdır. `best.pt` ağırlığının bağımsız değerlendirmesi veya saha testi değildir:

| Deney | Kayıt | Son satırdaki ölçüm |
|---|---|---|
| Konteyner sınıflandırma | 20 epoch | Top-1 accuracy: 0.33333 |
| Konteyner detection | 20 epoch | mAP50: 0.74625; mAP50-95: 0.73379 |
| CeyMo 20 epoch pilot | 20 epoch | Precision, recall ve mAP: 0 |
| Aktif Mapillary `train_advertisement_fix` | 2 epoch | Kutu mAP50: 0.08752; maske mAP50: 0.05624 |

Konteyner detection sayıları çok küçük veri ve tam-görsel bootstrap kutuları nedeniyle gerçek sokak görüntüsündeki nesne bulma başarımına genellenemez. Üç sınıflı sınıflandırmada top-5 değerinin 1 olması da ayırt edici bir başarı göstergesi değildir. Genel sistem doğruluğu, gizlilik recall'u ve gerçek zaman performansı için doğrulanmış tek bir toplam başarı oranı yoktur.

## 9. Teknik borç ve doğrulama eksikleri

- `TECHNICAL_AUDIT.md` geçmiş kısa MP4 testlerini raporlar; güncel kod için yeniden yapılmış test sayılmamalıdır. Rapordaki seri `HTTPServer` açıklaması artık güncel değildir.
- Uzun video, VFR/PTS zamanlama, döndürülmüş telefon videosu, bozuk codec ve farklı tarayıcı koşulları için kapsamlı doğrulama yoktur.
- Kod `cgi.FieldStorage` kullanır; kurulumda Python sürümü uyumluluğu doğrulanmalıdır.
- Bağımlılıklar kilitli değildir; model yolları ve çalışma dizini varsayımları taşınabilirliği azaltır.
- Veri hazırlamada rota/gün/video bazlı ayrım dokümante edilmiştir; bazı scriptlerde rastgele görsel veya dosya adına dayalı ayrım vardır. Gerçek veri sızıntısı denetlenmelidir.
- Model yükleme ve bazı hazırlıklar video döngüsündeki hata temizleme bloğundan önce yapılır; tüm hata yollarında temizliğin garanti edildiği varsayılmamalıdır.
- Depoda kaynak kodla birlikte sürdürülen kapsamlı bir otomatik test paketi görülmemiştir.
- Kullanıcı kimlik doğrulaması, yetkilendirme, kalıcı kayıt, arka plan iş yönetimi ve operasyonel izleme eksiktir.

## 10. Projeyi nasıl ele almalısın?

Projeyi “gerçek model çıkarımı içeren, veri ve saha doğrulaması sınırlı, belediye kullanımına yönelik yerel demo” olarak değerlendir. Tamamlanmış ürün veya üretime hazır denetim sistemi olarak tanımlama.

Öncelik önerirken şu sırayı temel al:

1. Gerçek rota verisinden, eğitimden ayrılmış test kümesi oluştur; her modülün yanlış pozitiflerini ve kaçırmalarını ölç.
2. Konteynerler için gerçek nesne kutuları topla; çoklu konteyner durumunu ayrı kırpıntılar üzerinden değerlendirecek akışı tasarla.
3. Taşan çöp anotasyonlarını tamamla; yol/kaldırım ilişkisini yüzey maskeleriyle doğrula.
4. Gerçek analiz sonuçlarını zaman, konum, kutu/maske, model sürümü, kaynak ve personel durumunu içeren ortak olay sözleşmesine bağla.
5. Sonuçları haritaya aktar; tekrar görülen aynı fiziksel sorunu tek olay altında birleştirmeyi tasarla.
6. Gizlilik modellerini ve bulanıklaştırma politikasını etiketli saha görüntülerinde ölç.
7. Video işlerini arka plan kuyruğuna taşı; ilerleme, iptal, kaynak sınırları ve tüm API uçlarında model eşzamanlılığını ele al.
8. Kalıcı kayıt, kullanıcı erişimi ve personel iş akışını ürün gereksinimleri netleştikçe ekle.

Bu adımlar öneridir; tamamlanmış işler değildir. Büyük mimari değişiklik veya uzun eğitim başlatmadan önce somut ihtiyacı ve beklenen kazanımı açıkla. İlk hedef mevcut sistemi anlaşılır ve ölçülebilir hale getirmektir.

## 11. Senden beklediğim ilk yanıt

Bu bağlama dayanarak Türkçe bir değerlendirme hazırla:

- Proje hangi sorunu, kim için ve hangi veri akışıyla çözmek istiyor?
- Kodda mevcut olan, yalnızca pilot düzeyde olan ve henüz plan aşamasında olan özellikler hangileri?
- En kritik veri, model, gizlilik ve uygulama eksikleri neler?
- İlk geliştirme iterasyonunda hangi somut çıktılar hedeflenmeli ve bunlar nasıl doğrulanmalı?

Her önemli tespiti mümkünse dosya veya ölçüm kaydıyla ilişkilendir. Gözlem, çıkarım ve öneriyi ayır. Yeni test yapmadıysan “test edildi” deme. Model güvenini başarı oranı, piksel ölçüsünü metre ölçüsü, örnek harita kayıtlarını gerçek saha bulgusu veya nesne tespitini kesin ihlal olarak sunma. Açıkça kod değişikliği istenmedikçe ilk yanıtında değerlendirme ve uygulanabilir önceliklendirme sun.

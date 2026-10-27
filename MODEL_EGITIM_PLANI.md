# Ayrıntılı model eğitimi planı

## Sınıflar

İlk eğitim sürümünde dört ayrı çıktı ailesi tutulacak:

- `road_damage`: çukur, boyuna çatlak, enine çatlak, ağ çatlağı
- `container_state`: konteyner, `empty`, `full`, `overflowing`, `occluded`
- `street_obstruction`: sandalye, masa, ürün yığını, kaldırım işgali alanı
- `advertising_board`: reklam panosu, dükkân tabelası, afiş

`occluded` ve `unknown` sınıfları, görünmeyen konteyner için zorunlu tahmin üretmemek amacıyla kullanılacak.

## Veri kuralı

Eğitim, doğrulama ve test görüntüleri video/gün/rota bazında ayrılacak. Aynı videodan art arda kareler farklı kümelere dağıtılmayacak. Yüz ve plakalar anotasyon öncesi bulanıklaştırılacak. Test kümesi eğitime kesinlikle girmeyecek.

## Başlangıç hedefi

Konteyner taşması, sokak işgali ve reklam panosu için ilk hedef sayıdan önce ölçülebilirliktir. Yol hasarında mevcut RDD2022 modeli referans olarak tutulacak; yerel görüntüler geldiğinde ayrıca kalibrasyon yapılacak.

## Eğitim çıktısı

Her modül aynı sözleşmeyle sonuç döndürecek: tür, güven, kutu/alan, zaman, konum, kaynak kare ve durum. Böylece model değişse bile web sitesi değişmeyecek.


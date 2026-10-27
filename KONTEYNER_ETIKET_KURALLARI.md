# Konteyner etiket kuralları

İlk sürümde üç sınıf kullanılacak:

- `empty`: Konteyner görünür ve içinde belirgin atık yok.
- `full`: Konteyner görünür, içinde atık var ancak atık konteynerin dışına taşmıyor.
- `overflowing`: Atık konteynerin dışına taşıyor veya konteyner çevresine yayılmış.

Bu nedenle `full`, "taşmamış ve boş değil" durumudur.

Konteynerin içi görünmüyorsa, görüntü çok uzaksa veya önünü araç/eşya kapatıyorsa bu görüntü üç sınıftan birine zorla sokulmayacak; veri toplama sırasında ayrılacak ve daha sonra `unknown` sınıfı olarak değerlendirilecek.

Örnek karar sırası:

```text
Dışarı taşan atık var mı?  -> overflowing
Hayır
İçeride atık görünüyor mu? -> full
Hayır                       -> empty
Görüntü yeterli mi?         -> değilse unknown/adaya alma
```

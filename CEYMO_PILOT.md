# CeyMo küçük pilot

- İndirilen alt küme: 100 eğitim görseli + 1 doğrulama görseli
- Eğitim: YOLO11n, 320 px, 3 epoch, CPU
- Ağırlık: `runs/detect/runs/ceymo_small/train_cpu/weights/best.pt`
- Pilot doğrulama mAP: 0.00 (tek doğrulama görseli ve 3 epoch, kalite ölçümü olarak kullanılmamalı)

Bu çalışma yalnızca veri formatını ve eğitim akışını doğrular. Şerit/işaretleme modelini demo kararlarına bağlamadan önce en az birkaç yüz görselle daha uzun eğitim ve ayrı test kümesi gerekir.

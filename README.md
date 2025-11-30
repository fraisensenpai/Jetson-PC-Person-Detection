# Jetson-PC-Person-Detection
Gerçek zamanlı insan tespiti yapan Jetson ve PC uyumlu YOLOv8 projesi.

# Jetson / PC YOLOv8 Person Detection

Bu proje, YOLOv8 kullanarak gerçek zamanlı insan tespiti yapar.  
Kod hem Jetson CSI kamerayla hem de USB kamerayla çalışacak şekilde hazırdır.

---

## ⚠️ Model Dosyası

Proje **`yolov8n.pt`** model dosyasına ihtiyaç duyar.  
Bu dosya GitHub’da depoya dahil edilmediği için manuel olarak indirip proje klasörüne koymalısınız:

- [YOLOv8n Model (Resmi)](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt)

Model boyutu ~14 MB olmalıdır.

---

## 🛠️ Kurulum

Python 3.10+ önerilir.

1. Paketleri kur:
```bash
pip install -r requirements.txt

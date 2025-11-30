import cv2
import os
import sys
import platform
from ultralytics import YOLO

# ---------------- Model Kontrol ----------------
MODEL_NAME = "yolov8n.pt"
MODEL_SIZE_MIN = 6 * 1024 * 1024 

if not os.path.exists(MODEL_NAME):
    print(f"{MODEL_NAME} bulunamadı! Lütfen manuel olarak indirip proje klasörüne koyun.")
    sys.exit()
elif os.path.getsize(MODEL_NAME) < MODEL_SIZE_MIN:
    print(f"{MODEL_NAME} eksik veya bozuk! Lütfen yeniden indirip proje klasörüne koyun.")
    sys.exit()
else:
    print(f"{MODEL_NAME} mevcut ve sağlıklı.")

# ---------------- Model Yükleme ----------------
model = YOLO(MODEL_NAME)
model.fuse()
person_id = 0 

# ---------------- Kamera Seçimi ----------------
USE_CSI = False
if platform.system() != "Windows":
    try:
        pipeline = ("nvarguscamerasrc ! "
                    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! "
                    "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
                    "videoconvert ! video/x-raw, format=BGR ! appsink")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            USE_CSI = True
            print("CSI kamera açıldı.")
    except:
        pass

if not USE_CSI:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        sys.exit()
    print("USB kamera kullanılıyor.")

# ---------------- Tespit Döngüsü ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model.predict(frame, conf=0.4, classes=[person_id])

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    title = "Person Detection (CSI)" if USE_CSI else "Person Detection (USB)"
    cv2.imshow(title, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

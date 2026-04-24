from ultralytics import YOLO
import cv2
import winsound
import time

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "fire_model.pt"
VIDEO_PATH = "input8.mp4"
CONF_THRESHOLD = 0.25
IMG_SIZE = 500   # Balanced size (better than 320)

# ==========================
# LOAD MODEL
# ==========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

cooldown = 0
prev_time = time.time()

print("🔥 Fire & Smoke Detection Running (Optimized CPU Mode)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to 1/4 area (efficient)
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Run inference (confidence handled inside model)
    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    fire_detected = False

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

            # Faster tuple check
            if class_name in ("fire", "fire-smoke", "factory-smoke", "fog"):

                fire_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (0, 0, 255) if "fire" in class_name else (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{class_name.upper()} {confidence:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    # Alarm logic
    if fire_detected and cooldown == 0:
        winsound.Beep(2000, 300)
        cooldown = 20

    if cooldown > 0:
        cooldown -= 1

    # Accurate FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("🔥 Fire & Smoke Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
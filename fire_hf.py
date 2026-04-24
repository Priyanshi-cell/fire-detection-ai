from ultralytics import YOLO
import cv2
import pygame
import time

# ==========================
# config
# ==========================
MODEL_PATH = "fire_model.pt"
VIDEO_PATH = "fireVid.mp4"

CONF_THRESHOLD = 0.25
IMG_SIZE = 416
COOLDOWN_FRAMES = 30

FIRE_SOUND_PATH = "fire_alarm.wav"
SMOKE_SOUND_PATH = "smoke_alert.wav"

# ==========================
# initialization
# ==========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

pygame.mixer.init()
fire_sound = pygame.mixer.Sound(FIRE_SOUND_PATH)
smoke_sound = pygame.mixer.Sound(SMOKE_SOUND_PATH)

fire_sound.set_volume(1.0)
smoke_sound.set_volume(0.6)

fire_cooldown = 0
smoke_cooldown = 0

prev_time = time.time()

print("🔥 Fire & Smoke Detection Started (Fire Priority Mode)")
print("Press 'q' to exit")

# ==========================
# main loop
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to 1/4 area
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    fire_detected = False
    smoke_detected = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

            # 🔥 STRICT FIRE ONLY
            if class_name == "fire":
                fire_detected = True

            # 🌫 STRICT SMOKE TYPES
            elif class_name in ("fire-smoke", "factory-smoke", "fog"):
                smoke_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw correct color
            if class_name == "fire":
                color = (0, 0, 255)  # Red
            else:
                color = (255, 0, 0)  # Blue

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

    # ==========================
    # prioority alarm
    # ==========================

    # 🔥 Fire has priority (only real fire)
    if fire_detected and fire_cooldown == 0:
        fire_sound.play()
        fire_cooldown = COOLDOWN_FRAMES
        smoke_cooldown = COOLDOWN_FRAMES  # suppress smoke

    # 🌫 Smoke only if NO fire
    elif smoke_detected and smoke_cooldown == 0:
        smoke_sound.play()
        smoke_cooldown = COOLDOWN_FRAMES

    # Cooldown counters
    if fire_cooldown > 0:
        fire_cooldown -= 1

    if smoke_cooldown > 0:
        smoke_cooldown -= 1

    # ==========================
    # FPS DISPLAY
    # ==========================
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

    cv2.imshow("🔥 Fire & Smoke Detection (Priority Mode)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
import cv2
import numpy as np
import winsound

# Start webcam
cap = cv2.VideoCapture(0)

# Red color ranges (important: red has 2 HSV ranges)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

alert_flag = False
cooldown_frames = 100
current_cooldown = cooldown_frames
min_contour_area = 1500


while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            fire_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "FIRE DETECTED!", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 🔔 Alarm part (Beep sound)
    if fire_detected and not alert_flag and current_cooldown == cooldown_frames:
        winsound.Beep(1500, 800)  # frequency, duration
        print("Fire Detected!")
        alert_flag = True

    # Cooldown system
    if alert_flag:
        current_cooldown -= 1
        if current_cooldown <= 0:
            alert_flag = False
            current_cooldown = cooldown_frames

    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
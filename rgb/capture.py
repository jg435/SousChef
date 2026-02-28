import cv2
import os
import time

DATA_DIR = "../data"

os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting capture... Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed")
            break

        timestamp = int(time.time())
        filename = os.path.join(DATA_DIR, f"rgb_{timestamp}.jpg")

        cv2.imwrite(filename, frame)
        print("Saved", filename)

        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopping capture.")

cap.release()

import time
import os
import numpy as np
import cv2
import board
import busio
import adafruit_mlx90640

# ---------- Settings ----------
SAVE_DIR = "/home/pi/souschef/data/sync"
CAPTURE_INTERVAL = 0.25  # seconds (4 fps, matches thermal camera)

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Setup Thermal ----------
i2c = busio.I2C(board.SCL, board.SDA)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
thermal_buffer = [0] * 768

# ---------- Setup RGB ----------
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError("Could not open RGB camera at /dev/video0")

print("Starting synchronized capture at 4 fps. Press Ctrl+C to stop.")
print(f"Saving to: {SAVE_DIR}")

try:
    while True:
        timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}ms"

        # --- Capture Thermal ---
        try:
            mlx.getFrame(thermal_buffer)
            arr = np.array(thermal_buffer, dtype=np.float32).reshape((24, 32))

            t_min = float(np.min(arr))
            t_max = float(np.max(arr))
            denom = (t_max - t_min) if (t_max - t_min) > 1e-6 else 1.0
            norm = ((arr - t_min) / denom * 255.0).astype(np.uint8)

            vis = cv2.resize(norm, (32 * 15, 24 * 15), interpolation=cv2.INTER_NEAREST)
            thermal_color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)

            cv2.putText(thermal_color, f"min {t_min:.1f}C", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(thermal_color, f"max {t_max:.1f}C", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            thermal_path = os.path.join(SAVE_DIR, f"thermal_{timestamp}.jpg")
            cv2.imwrite(thermal_path, thermal_color)

            # Also save raw temperature array as .npy for later analysis
            raw_path = os.path.join(SAVE_DIR, f"thermal_raw_{timestamp}.npy")
            np.save(raw_path, arr)

        except Exception as e:
            print(f"Thermal capture error: {e}")
            time.sleep(CAPTURE_INTERVAL)
            continue

        # --- Capture RGB ---
        ret, frame = cap.read()
        if ret:
            rgb_path = os.path.join(SAVE_DIR, f"rgb_{timestamp}.jpg")
            cv2.imwrite(rgb_path, frame)
        else:
            print("RGB frame capture failed")

        # --- Composite (side-by-side) ---
        if ret:
            rgb_h, rgb_w = frame.shape[:2]   # 720, 1280

            # Scale thermal to match RGB height, preserve aspect (24:32 = 3:4)
            th_h = rgb_h                     # 720
            th_w = int(th_h * 32 / 24)       # 960

            th_resized = cv2.resize(thermal_color, (th_w, th_h), interpolation=cv2.INTER_NEAREST)

            # Labels
            label_args = (cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame,      "RGB",     (10, 30), *label_args)
            cv2.putText(th_resized, "THERMAL", (10, 30), *label_args)

            composite = np.hstack([frame, th_resized])
            comp_path = os.path.join(SAVE_DIR, f"composite_{timestamp}.jpg")
            cv2.imwrite(comp_path, composite)

        print(f"Saved pair: {timestamp}  |  thermal {t_min:.1f}-{t_max:.1f}°C")

        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    print("\nStopping capture.")

finally:
    cap.release()
    print("Camera released. Done.")
    
import time
import board
import busio
import adafruit_mlx90640

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)

# Lower refresh rate is more stable initially
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

frame = [0] * 768

print("Reading one frame...")
mlx.getFrame(frame)

t_min = min(frame)
t_max = max(frame)
t_avg = sum(frame) / len(frame)

print(f"min={t_min:.2f}C max={t_max:.2f}C avg={t_avg:.2f}C")

"""
APA102 / DotStar NeoPixel test.
Wiring: VCC→Pin2(5V), GND→Pin6, C0→Pin23(SPI CLK), D0→Pin19(SPI MOSI)
Requires: dtparam=spi=on in /boot/firmware/config.txt + reboot
"""
import time
import board
import adafruit_dotstar as dotstar

NUM_PIXELS = 10

pixels = dotstar.DotStar(board.SCLK, board.MOSI, NUM_PIXELS, brightness=0.3, baudrate=1000000)

print(f"Testing {NUM_PIXELS} APA102 pixels on SPI0 (GPIO 10/11)...")

# Test 1: red, green, blue in sequence
for colour, name in [((255, 0, 0), "RED"), ((0, 255, 0), "GREEN"), ((0, 0, 255), "BLUE")]:
    print(f"  All {name}")
    pixels.fill(colour)
    pixels.show()
    time.sleep(1)

# Test 2: chase
print("  White chase...")
pixels.fill((0, 0, 0))
for _ in range(2):
    for i in range(NUM_PIXELS):
        pixels[i] = (255, 255, 255)
        pixels.show()
        time.sleep(0.1)
        pixels[i] = (0, 0, 0)

# Test 3: all off
pixels.fill((0, 0, 0))
pixels.show()
print("Done — pixels off.")

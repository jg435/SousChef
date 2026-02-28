"""
Speaker test for CQRobot 3W 4Ohm on GPIO 18, Raspberry Pi 4B.

Findings:
  - audremap overlay does NOT route snd_bcm2835 audio to GPIO 18 on BCM2711 (Pi 4).
    It only works on BCM2835 (Pi 3/Zero). GPIO 18 stays 'input' during ALSA playback.
  - Raw PWM on GPIO 18 via sysfs DOES work (pwmchip0/pwm0 → Alt5 → PWM0_0).
  - snd_bcm2835 audio goes to GPIO 40/41 (3.5mm jack circuit). Plug into that jack
    to confirm audio software works, then use a PAM8403 amplifier board on GPIO 18
    for full speaker output.

Tests in this file:
  1. ALSA / espeak-ng → 3.5mm jack  (confirms audio pipeline works)
  2. Raw PWM tone on GPIO 18         (confirms pin is physically connected)
"""
import os
import subprocess
import sys
import time

ESPEAK_SPEED = 140
ESPEAK_AMP   = 200
PWM_CHIP     = "/sys/class/pwm/pwmchip0"
PWM_DEV      = PWM_CHIP + "/pwm0"


# ── helpers ──────────────────────────────────────────────────────────────────

def run(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"  ERROR: {' '.join(cmd)}\n  {r.stderr.strip()}")
        sys.exit(1)
    return r

def gpio_state(pin):
    r = subprocess.run(["pinctrl", "get", str(pin)], capture_output=True, text=True)
    return r.stdout.strip()

def pwm_write(path, value):
    with open(path, "w") as f:
        f.write(str(value))


# ── Test 1: ALSA / espeak → 3.5mm jack ───────────────────────────────────────

print("=" * 60)
print("TEST 1: ALSA audio via 3.5mm jack (snd_bcm2835, GPIO 40/41)")
print("  Connect headphones or a speaker to the 3.5mm jack to verify.")
print("=" * 60)

# Set volume to 100%
run(["amixer", "-q", "-M", "sset", "PCM", "100%"])
r = run(["amixer", "-M", "sget", "PCM"], check=False)
for line in r.stdout.splitlines():
    if "%" in line:
        print(f"  Volume: {line.strip()}")

print("  Playing 440 Hz tone for 1 second via ALSA...")
run(["speaker-test", "-t", "sin", "-f", "440", "-c", "1", "-l", "1",
     "-s", "1"], check=False)

print("  Speaking via espeak-ng → sox +20dB → aplay hw:0,0 ...")
import shlex
espeak = subprocess.Popen(
    ["espeak-ng", "--stdout", "-s", str(ESPEAK_SPEED),
     "SousChef audio test. If you hear this, the pipeline is working."],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
)
sox = subprocess.Popen(
    ["sox", "-t", "wav", "-", "-t", "wav", "-", "gain", "20"],
    stdin=espeak.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
)
espeak.stdout.close()
aplay = subprocess.Popen(
    ["aplay", "-D", "hw:0,0", "-"],
    stdin=sox.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
sox.stdout.close()
aplay.wait()

print()


# ── Test 2: Raw PWM tone on GPIO 18 ──────────────────────────────────────────

print("=" * 60)
print("TEST 2: Raw PWM tone on GPIO 18 (1 kHz, 50% duty cycle)")
print("  You should hear a faint 1 kHz buzz if the speaker is wired.")
print("  NOTE: GPIO 18 sources max ~16 mA. A 4-ohm speaker needs an")
print("  amplifier (e.g. PAM8403) for audible output.")
print("=" * 60)

if not os.path.exists(PWM_CHIP):
    print("  FAIL: pwmchip0 not found. PWM overlay not loaded.")
    sys.exit(1)

# Export PWM0
if not os.path.exists(PWM_DEV):
    try:
        pwm_write(PWM_CHIP + "/export", 0)
        time.sleep(0.2)
    except PermissionError:
        print("  FAIL: cannot export pwm0 (run with sudo?)")
        sys.exit(1)

print(f"  GPIO 18 before enable: {gpio_state(18)}")

# 1 kHz tone: period = 1,000,000 ns, 50% duty
pwm_write(PWM_DEV + "/duty_cycle", 0)
pwm_write(PWM_DEV + "/period", 1_000_000)
pwm_write(PWM_DEV + "/duty_cycle", 500_000)
pwm_write(PWM_DEV + "/enable", 1)

print(f"  GPIO 18 after  enable: {gpio_state(18)}")
print("  Buzzing for 2 seconds...")
time.sleep(2)

pwm_write(PWM_DEV + "/enable", 0)
pwm_write(PWM_DEV + "/duty_cycle", 0)
pwm_write(PWM_CHIP + "/unexport", 0)

print(f"  GPIO 18 after cleanup: {gpio_state(18)}")
print()


# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 60)
print("SUMMARY")
print("  audremap does not work on Pi 4 (BCM2711) for GPIO 18.")
print("  To get espeak audio from GPIO 18 speaker:")
print("  1. Add a PAM8403 (or similar) amplifier between GPIO 18 and speaker.")
print("  2. Route audio: dtoverlay=pwm-2chan will drive the amp input.")
print("  OR plug speaker into the 3.5mm jack directly (no amp needed).")
print("=" * 60)

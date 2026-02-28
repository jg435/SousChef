"""
Speaker test for CQRobot 3W 4Ohm on GPIO 18 (Pi 4 bcm2835 PWM audio).

GPIO 18 is the bcm2835 PWM0 output — driven by dtparam=audio=on.
Audio card: 'bcm2835 Headphones' (card 0, mono).
"""
import subprocess
import sys
import time


ESPEAK_SPEED = 140   # words per minute (try 120-160)
ESPEAK_AMP   = 200   # amplitude 0-200


def run(cmd, check=True):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"ERROR running {' '.join(cmd)}:\n{result.stderr.strip()}")
        sys.exit(1)
    return result


def show_audio_info():
    print("=== ALSA devices ===")
    r = run(["aplay", "-l"], check=False)
    for line in r.stdout.splitlines():
        if "bcm2835" in line or "card" in line.lower():
            print(" ", line)

    print("\n=== PCM volume ===")
    r = run(["amixer", "-M", "sget", "PCM"], check=False)
    for line in r.stdout.splitlines():
        if any(k in line for k in ("Mono", "Front", "dB", "%")):
            print(" ", line.strip())


def set_volume(pct):
    print(f"\n>>> Setting PCM volume to {pct}%")
    run(["amixer", "-q", "-M", "sset", "PCM", f"{pct}%"])


def speak(text):
    print(f"    Speaking: \"{text}\"")
    result = subprocess.run(
        ["espeak-ng", "-s", str(ESPEAK_SPEED), "-a", str(ESPEAK_AMP), text],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    espeak-ng error: {result.stderr.strip()}")
    time.sleep(0.5)


# ── Run tests ────────────────────────────────────────────────────────────────

show_audio_info()

print("\n=== Test 1: volume at 75% ===")
set_volume(75)
speak("SousChef speaker test at 75 percent.")

print("\n=== Test 2: volume at 100% ===")
set_volume(100)
speak("SousChef speaker test at 100 percent.")

print("\n=== Test 3: kitchen phrase ===")
speak("Your pan is hot and ready for oil.")

print("\n=== Test 4: warning phrase ===")
speak("Warning — the onions are starting to burn.")

print("\nDone. Leaving volume at 100%.")

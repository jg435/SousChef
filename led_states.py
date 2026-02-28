"""
LED state animations for 10-pixel APA102 (DotStar) strip.
Wiring: VCC→5V, GND→GND, C0→GPIO11 (SPI CLK), D0→GPIO10 (SPI MOSI)

Usage:
    import led_states
    led_states.start()
    led_states.set_led_state("COOKING")
"""
import math
import threading
import time
import logging

import board
import adafruit_dotstar as dotstar

NUM_PIXELS   = 10
_BAUD        = 1_000_000

_pixels: dotstar.DotStar | None = None
_state   = "NO_STOVE"
_lock    = threading.Lock()
_change  = threading.Event()   # set whenever state changes


# ── Public API ────────────────────────────────────────────────────────────────

def start():
    """Start the background animation thread. Call once at startup."""
    t = threading.Thread(target=_loop, daemon=True)
    t.start()


def set_led_state(state: str):
    """Switch to a new cooking state. Safe to call from any thread."""
    global _state
    with _lock:
        _state = state
    _change.set()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _px() -> dotstar.DotStar:
    global _pixels
    if _pixels is None:
        _pixels = dotstar.DotStar(
            board.SCLK, board.MOSI, NUM_PIXELS,
            brightness=0.8, baudrate=_BAUD,
        )
    return _pixels


def _changed() -> bool:
    return _change.is_set()


def _sleep(seconds: float) -> bool:
    """Sleep in small steps; return True early if state changes."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _changed():
            return True
        time.sleep(0.02)
    return False


# ── Animation loop ────────────────────────────────────────────────────────────

def _loop():
    while True:
        _change.clear()
        with _lock:
            state = _state
        try:
            _dispatch(state)
        except Exception as e:
            logging.error(f"LED animation error ({state}): {e}")
            _change.wait(timeout=1.0)


def _dispatch(state: str):
    px = _px()
    if state == "NO_STOVE":
        # White dim breathing, slow (capped at 0.2 brightness)
        _breathe(px, (255, 255, 255), low=0.02, high=0.2, period=3.0)

    elif state == "IDLE":
        # Blue solid, low brightness
        px.brightness = 0.3
        px.fill((0, 0, 255))
        px.show()
        _change.wait()

    elif state == "PREHEATING":
        # Orange slow chase, one pixel at a time
        _chase(px, (255, 140, 0), brightness=0.8, step=0.2)

    elif state == "READY":
        # 3 quick green flashes, then solid green
        _flash_then_solid(px, (0, 255, 0), brightness=0.8, flashes=3, duration=0.15)

    elif state == "COOKING":
        # Green slow breathing
        _breathe(px, (0, 200, 0), low=0.1, high=0.8, period=2.0)

    elif state == "DONE":
        # Green + white alternating chase
        _alt_chase(px, (0, 220, 0), (255, 255, 255), brightness=0.8, step=0.15)

    elif state == "OVERDONE":
        # Red fast strobe + console warning each cycle
        _strobe(px, (255, 0, 0), brightness=0.8, on=0.08, off=0.08)

    else:
        px.fill((0, 0, 0))
        px.show()
        _change.wait()


# ── Animation primitives ──────────────────────────────────────────────────────

def _breathe(px, color, low, high, period):
    steps = 60
    step_time = period / steps
    while not _changed():
        for i in range(steps):
            if _changed():
                return
            t = i / steps
            b = low + (high - low) * (0.5 - 0.5 * math.cos(2 * math.pi * t))
            px.brightness = b
            px.fill(color)
            px.show()
            if _sleep(step_time):
                return


def _chase(px, color, brightness, step):
    px.brightness = brightness
    while not _changed():
        for i in range(NUM_PIXELS):
            if _changed():
                return
            px.fill((0, 0, 0))
            px[i] = color
            px.show()
            if _sleep(step):
                return


def _flash_then_solid(px, color, brightness, flashes, duration):
    px.brightness = brightness
    for _ in range(flashes):
        if _changed():
            return
        px.fill(color)
        px.show()
        if _sleep(duration):
            return
        px.fill((0, 0, 0))
        px.show()
        if _sleep(duration):
            return
    # Hold solid until next state change
    px.fill(color)
    px.show()
    _change.wait()


def _alt_chase(px, color_a, color_b, brightness, step):
    px.brightness = brightness
    offset = 0
    while not _changed():
        for i in range(NUM_PIXELS):
            px[i] = color_a if (i + offset) % 2 == 0 else color_b
        px.show()
        offset ^= 1
        if _sleep(step):
            return


def _strobe(px, color, brightness, on, off):
    px.brightness = brightness
    flash_count = 0
    while not _changed():
        px.fill(color)
        px.show()
        if _sleep(on):
            break
        px.fill((0, 0, 0))
        px.show()
        if _sleep(off):
            break
        flash_count += 1
        # Log once per second (~6 flashes at 80ms on+off)
        if flash_count % 6 == 0:
            logging.warning("OVERDONE — food may be burning!")
    px.fill((0, 0, 0))
    px.show()

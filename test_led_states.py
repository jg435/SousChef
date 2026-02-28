import time
import led_states

led_states.start()

states = [
    ("NO_STOVE",    3, "White dim breathing"),
    ("IDLE",        3, "Blue solid"),
    ("PREHEATING",  4, "Orange chase"),
    ("READY",       4, "3 flashes → solid green"),
    ("COOKING",     4, "Green breathing"),
    ("DONE",        4, "Green/white alt chase"),
    ("OVERDONE",    4, "Red strobe (watch console for warnings)"),
]

for state, duration, description in states:
    print(f"[{state}] {description}")
    led_states.set_led_state(state)
    time.sleep(duration)

led_states.set_led_state("IDLE")
print("Done.")

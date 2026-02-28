#!/usr/bin/env python3
"""SousChef interactive cooking assistant simulator.

A terminal REPL that simulates the real Pi system using photos
from disk instead of live cameras.

Usage:
    python simulate.py

Commands:
    load <path>          — set the current camera frame from a file
    temp <min> <max> <avg> — set mock thermal readings (°C)
    help                 — show commands
    quit / exit          — exit
    <anything else>      — ask a natural language question
"""

import base64
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "google/gemini-2.5-flash"
HISTORY_LIMIT = 6  # number of recent exchanges to include as context

VALID_STATES = ("NO_STOVE", "IDLE", "PREHEATING", "READY", "COOKING", "DONE", "OVERDONE")

PROACTIVE_SYSTEM = """\
You are SousChef, a kitchen assistant watching through a camera above a stovetop.

Determine the current cooking state and share a brief observation.

States (in order of typical progression):
- NO_STOVE: Image does NOT show a stove or cooktop. Camera may be misaligned or obstructed.
- IDLE: Stove visible but off. No active heat, no food.
- PREHEATING: Burner on, pan/pot warming up, not hot enough to cook yet.
- READY: Pan/pot is hot and ready for food (heat shimmer, oil spreading).
- COOKING: Food is actively being cooked.
- DONE: Food appears fully cooked and burner is off. Ready to plate.
- OVERDONE: Food is overcooked — burning, charred, dried out, or smoking. Needs immediate attention.

Format your response EXACTLY as:
STATE: <one of the states above>
<your 1-2 sentence observation>

Example:
STATE: COOKING
The egg whites are starting to set around the edges — don't touch it yet.\
"""

REACTIVE_SYSTEM = (
    "You are SousChef, a warm and knowledgeable cooking assistant "
    "watching through a kitchen camera. You also have a thermal sensor. "
    "The current cooking state and recent state history are provided for context. "
    "Give brief, practical, conversational answers — 1 to 3 sentences. "
    "Speak like a helpful chef standing next to the cook."
)


def encode_image(path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, mime_type)."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def make_image_content(img_b64: str, mime: str) -> dict:
    """Build an OpenAI-style image_url content block."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{img_b64}"},
    }


def thermal_text(thermal: dict | None) -> str:
    """Format thermal readings as a text string, or empty if unset."""
    if thermal is None:
        return ""
    return (
        f"Thermal sensor — min: {thermal['min']:.1f}°C, "
        f"max: {thermal['max']:.1f}°C, avg: {thermal['avg']:.1f}°C."
    )


def build_history_messages(history: list[dict]) -> list[dict]:
    """Return the last HISTORY_LIMIT messages from conversation history."""
    return history[-HISTORY_LIMIT:]


def parse_state_response(raw: str) -> tuple[str | None, str]:
    """Parse a 'STATE: ...\nobservation' response into (state, observation)."""
    lines = raw.strip().splitlines()
    if lines and lines[0].startswith("STATE:"):
        state = lines[0].split(":", 1)[1].strip().upper()
        observation = "\n".join(lines[1:]).strip()
        if state not in VALID_STATES:
            state = None
        return state, observation
    return None, raw.strip()


def state_history_text(state_log: list[dict]) -> str:
    """Format recent state transitions with observations for LLM context."""
    if not state_log:
        return ""
    lines = ["State history:"]
    for entry in state_log[-5:]:
        lines.append(f"- {entry['state']}: {entry['observation']}")
    return "\n".join(lines)


def call_proactive(client: OpenAI, img_b64: str, mime: str,
                   thermal: dict | None, history: list[dict],
                   state_log: list[dict]) -> tuple[str | None, str]:
    """Proactive call: auto-analyze a scene. Returns (state, observation)."""
    user_content = [make_image_content(img_b64, mime)]

    text_parts = []
    t = thermal_text(thermal)
    if t:
        text_parts.append(t)
    sh = state_history_text(state_log)
    if sh:
        text_parts.append(sh)
    text_parts.append("What do you observe?")
    user_content.append({"type": "text", "text": "\n".join(text_parts)})

    messages = build_history_messages(history) + [
        {"role": "user", "content": user_content},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": PROACTIVE_SYSTEM}] + messages,
    )
    return parse_state_response(response.choices[0].message.content)


def call_reactive(client: OpenAI, question: str, img_b64: str | None,
                  mime: str | None, thermal: dict | None,
                  history: list[dict], current_state: str | None,
                  state_log: list[dict]) -> str:
    """Reactive call: answer a user question with current context."""
    user_content = []

    if img_b64 and mime:
        user_content.append(make_image_content(img_b64, mime))

    text_parts = []
    t = thermal_text(thermal)
    if t:
        text_parts.append(t)
    if current_state:
        text_parts.append(f"Current state: {current_state}")
    sh = state_history_text(state_log)
    if sh:
        text_parts.append(sh)
    text_parts.append(f"Question: {question}")
    user_content.append({"type": "text", "text": "\n".join(text_parts)})

    messages = build_history_messages(history) + [
        {"role": "user", "content": user_content},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": REACTIVE_SYSTEM}] + messages,
    )
    return response.choices[0].message.content.strip()


HELP_TEXT = """
Commands:
  load <path>              Load an image as the current camera frame
  temp <min> <max> <avg>   Set mock thermal readings (°C)
  help                     Show this help message
  quit / exit              Exit the simulator

Anything else is treated as a question to the cooking assistant.
""".strip()


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set. Add it to .env or export it.",
              file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # State
    current_image: tuple[str, str] | None = None  # (b64, mime)
    current_image_path: str | None = None
    thermal: dict | None = None  # {"min": float, "max": float, "avg": float}
    history: list[dict] = []  # conversation history
    current_state: str | None = None
    state_log: list[dict] = []  # [{"state": ..., "observation": ...}, ...]

    print("SousChef Simulator")
    print("Type 'help' for commands.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not line:
            continue

        # --- quit / exit ---
        if line.lower() in ("quit", "exit"):
            print("Bye!")
            break

        # --- help ---
        if line.lower() == "help":
            print(HELP_TEXT)
            continue

        # --- load <path> ---
        if line.lower().startswith("load "):
            path = line[5:].strip()
            if not os.path.isfile(path):
                print(f"File not found: {path}")
                continue
            try:
                img_b64, mime = encode_image(path)
            except Exception as e:
                print(f"Error reading image: {e}")
                continue

            current_image = (img_b64, mime)
            current_image_path = path
            print(f"Loaded: {path}")

            # Proactive analysis
            try:
                state, observation = call_proactive(
                    client, img_b64, mime, thermal, history, state_log,
                )
                if state:
                    current_state = state
                    state_log.append({"state": state, "observation": observation})
                    print(f"[{state}] {observation}")
                else:
                    print(observation)
                history.append({"role": "user", "content": f"[loaded image: {path}]"})
                history.append({"role": "assistant", "content": observation})
            except Exception as e:
                print(f"LLM error: {e}")
            continue

        # --- temp <min> <max> <avg> ---
        if line.lower().startswith("temp "):
            parts = line.split()
            if len(parts) != 4:
                print("Usage: temp <min> <max> <avg>")
                continue
            try:
                t_min = float(parts[1])
                t_max = float(parts[2])
                t_avg = float(parts[3])
            except ValueError:
                print("Usage: temp <min> <max> <avg> (numbers)")
                continue

            thermal = {"min": t_min, "max": t_max, "avg": t_avg}
            print(f"Thermal set: min={t_min:.1f}°C  max={t_max:.1f}°C  avg={t_avg:.1f}°C")

            # If we have an image loaded, give proactive thermal feedback
            if current_image:
                img_b64, mime = current_image
                try:
                    state, observation = call_proactive(
                        client, img_b64, mime, thermal, history, state_log,
                    )
                    if state:
                        current_state = state
                        state_log.append({"state": state, "observation": observation})
                        print(f"[{state}] {observation}")
                    else:
                        print(observation)
                    history.append({
                        "role": "user",
                        "content": f"[thermal update: min={t_min}, max={t_max}, avg={t_avg}]",
                    })
                    history.append({"role": "assistant", "content": observation})
                except Exception as e:
                    print(f"LLM error: {e}")
            continue

        # --- natural language question ---
        img_b64 = current_image[0] if current_image else None
        mime = current_image[1] if current_image else None

        try:
            answer = call_reactive(
                client, line, img_b64, mime, thermal, history,
                current_state, state_log,
            )
            print(answer)
            history.append({"role": "user", "content": line})
            history.append({"role": "assistant", "content": answer})
        except Exception as e:
            print(f"LLM error: {e}")


if __name__ == "__main__":
    main()

from flask import Flask, Response, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
import base64
import edge_tts
import io
import json
import queue
import os
import sys
import threading
import time
import numpy as np
import cv2

if "--mock" not in sys.argv:
    import board
    import busio
    import adafruit_mlx90640

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

app = Flask(__name__)

SYNC_DIR = "/home/pi/souschef/data/sync"
if "--mock" not in sys.argv:
    os.makedirs(SYNC_DIR, exist_ok=True)

ai = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

# ---------- TTS (edge-tts → MP3 served to browser) ----------
def _tts_bytes(text, voice="en-US-AriaNeural", rate="+25%"):
    """Generate MP3 bytes from text using edge-tts."""
    async def _gen():
        buf = io.BytesIO()
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()
    return asyncio.run(_gen())


# ---------- SSE broadcast (proactive alerts → all connected browsers) ----------
_sse_listeners      = []
_sse_listeners_lock = threading.Lock()

def _broadcast(text):
    """Push a message to every connected SSE client."""
    with _sse_listeners_lock:
        for q in _sse_listeners:
            q.put(text)

# ---------- Shared state (written by capture thread, read by Flask) ----------
_lock  = threading.Lock()
_rgb     = None   # latest BGR frame (numpy)
_thermal = None   # latest colormap frame (numpy)
_composite = None
_t_min   = 0.0
_t_max   = 0.0
_t_avg   = 0.0

# ---------- State tracking (structured proactive analysis) ----------
_current_state = None
_state_log     = []

VALID_STATES = ("NO_STOVE", "IDLE", "PREHEATING", "READY", "COOKING", "DONE", "OVERDONE")

PROACTIVE_SYSTEM = """\
You are SousChef, a kitchen assistant watching through a camera above a stovetop.

Determine the current cooking state and share a brief observation.

States (in order of typical progression):
- NO_STOVE: Image does NOT show a stove or cooktop.
- IDLE: Stove visible but off. No active heat, no food.
- PREHEATING: Burner on, pan/pot warming up, not hot enough to cook yet.
- READY: Pan/pot is hot and ready for food.
- COOKING: Food is actively being cooked.
- DONE: Food appears fully cooked. Ready to plate.
- OVERDONE: Food is overcooked — burning, charred, or smoking. Needs immediate attention.

FEEDBACK rules — ONLY provide feedback when the cook is making a mistake:
- Heat too high/low for what they're cooking (say "lower the heat" or "try medium heat", never cite degrees)
- Food needs flipping/stirring and they haven't
- Something is burning or about to burn
- Pan is dry and food is sticking
- Leave FEEDBACK blank when things are going fine. No praise, no encouragement, no "looking good".

Never mention temperature numbers. Use terms like "too hot", "not hot enough", "medium heat", "low heat".

Format EXACTLY as:
STATE: <one of the states above>
OBS: <1 short sentence>
FEEDBACK: <correction or leave blank>\
"""


def parse_state_response(raw):
    """Parse structured LLM response into (state, observation, feedback)."""
    state = None
    observation = ""
    feedback = None
    for line in raw.strip().splitlines():
        if line.startswith("STATE:"):
            val = line.split(":", 1)[1].strip().upper()
            if val in VALID_STATES:
                state = val
        elif line.startswith("OBS:"):
            observation = line.split(":", 1)[1].strip()
        elif line.startswith("FEEDBACK:"):
            val = line.split(":", 1)[1].strip()
            if val:
                feedback = val
    if not observation:
        observation = raw.strip()
    return state, observation, feedback


def state_history_text(log):
    """Format recent state transitions for LLM context."""
    if not log:
        return ""
    lines = ["Recent state history:"]
    for entry in log[-5:]:
        line = f"- {entry['state']}: {entry['observation']}"
        if entry.get("feedback"):
            line += f" (feedback: {entry['feedback']})"
        lines.append(line)
    return "\n".join(lines)


# ---------- Capture thread ----------
def capture_loop():
    global _rgb, _thermal, _composite, _t_min, _t_max, _t_avg

    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        print("ERROR: could not open /dev/video0")
        return

    i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
    buf = [0] * 768

    print("Capture loop running at 4 fps. Saving to", SYNC_DIR)

    try:
        while True:
            ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}ms"

            try:
                mlx.getFrame(buf)
                arr = np.array(buf, dtype=np.float32).reshape((24, 32))
                t_min = float(np.min(arr))
                t_max = float(np.max(arr))
                t_avg = float(np.mean(arr))
                denom = (t_max - t_min) if (t_max - t_min) > 1e-6 else 1.0
                norm  = ((arr - t_min) / denom * 255.0).astype(np.uint8)
                vis   = cv2.resize(norm, (480, 360), interpolation=cv2.INTER_NEAREST)
                thermal_color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
                cv2.putText(thermal_color, f"min {t_min:.1f}C", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(thermal_color, f"max {t_max:.1f}C", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except ValueError:
                continue  # MLX90640 occasionally returns a corrupt frame; skip silently
            except Exception as e:
                print(f"Thermal error: {e}")
                continue

            ret, frame = cap.read()
            if not ret:
                print("RGB frame failed")
                continue
            frame = cv2.flip(frame, 1)

            rgb_h, rgb_w = frame.shape[:2]
            th_full = cv2.resize(thermal_color, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
            composite = cv2.addWeighted(frame, 0.6, th_full, 0.4, 0)
            cv2.putText(composite, f"min {t_min:.1f}C  max {t_max:.1f}C", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            np.save(os.path.join(SYNC_DIR, f"thermal_raw_{ts}.npy"), arr)
            cv2.imwrite(os.path.join(SYNC_DIR, f"rgb_{ts}.jpg"),       frame)
            cv2.imwrite(os.path.join(SYNC_DIR, f"thermal_{ts}.jpg"),   thermal_color)
            cv2.imwrite(os.path.join(SYNC_DIR, f"composite_{ts}.jpg"), composite)

            with _lock:
                _rgb       = frame.copy()
                _thermal   = thermal_color.copy()
                _composite = composite.copy()
                _t_min, _t_max, _t_avg = t_min, t_max, t_avg

            print(f"  {ts}  {t_min:.1f}-{t_max:.1f}°C  avg {t_avg:.1f}°C")

    finally:
        cap.release()
        print("Camera released.")


# ---------- Mock capture thread (reads from video file) ----------
def mock_capture_loop(video_path):
    global _rgb, _thermal, _composite, _t_min, _t_max, _t_avg

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1.0 / fps
    start = time.time()

    print(f"Mock capture running from {video_path} at {fps:.0f} fps")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                print("ERROR: could not read video frames")
                return

        elapsed = time.time() - start
        # Simulate temp ramp: 25°C → 200°C over 2 minutes, then hold
        base = 25 + min(elapsed / 120, 1.0) * 175
        t_min = base - 15 + np.random.normal(0, 2)
        t_max = base + 10 + np.random.normal(0, 2)
        t_avg = (t_min + t_max) / 2

        # Build fake thermal vis from frame brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (32, 24))
        thermal_arr = (small.astype(np.float32) / 255.0) * (t_max - t_min) + t_min
        denom = (t_max - t_min) if (t_max - t_min) > 1e-6 else 1.0
        norm = ((thermal_arr - t_min) / denom * 255.0).astype(np.uint8)
        vis = cv2.resize(norm, (480, 360), interpolation=cv2.INTER_NEAREST)
        thermal_color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        cv2.putText(thermal_color, f"min {t_min:.1f}C", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(thermal_color, f"max {t_max:.1f}C", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        rgb_h, rgb_w = frame.shape[:2]
        th_full = cv2.resize(thermal_color, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        composite = cv2.addWeighted(frame, 0.6, th_full, 0.4, 0)
        cv2.putText(composite, f"min {t_min:.1f}C  max {t_max:.1f}C", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        with _lock:
            _rgb = frame.copy()
            _thermal = thermal_color.copy()
            _composite = composite.copy()
            _t_min, _t_max, _t_avg = t_min, t_max, t_avg

        time.sleep(delay)

    cap.release()


# ---------- Proactive observation loop ----------
def proactive_loop():
    """Every 15 seconds, analyze the scene and broadcast structured feedback."""
    global _current_state

    while True:
        time.sleep(5)

        with _lock:
            frame   = _rgb.copy() if _rgb is not None else None
            t_min_v = _t_min
            t_max_v = _t_max
            t_avg_v = _t_avg

        if frame is None:
            continue

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            continue
        img_b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")

        text_parts = [
            f"Thermal sensor — min: {t_min_v:.1f}°C, max: {t_max_v:.1f}°C, avg: {t_avg_v:.1f}°C."
        ]
        sh = state_history_text(_state_log)
        if sh:
            text_parts.append(sh)
        text_parts.append("What do you observe?")

        try:
            msg = ai.chat.completions.create(
                model="anthropic/claude-opus-4.6",
                max_tokens=80,
                messages=[
                    {
                        "role": "system",
                        "content": PROACTIVE_SYSTEM
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                            },
                            {
                                "type": "text",
                                "text": "\n".join(text_parts)
                            }
                        ]
                    }
                ]
            )
            raw = msg.choices[0].message.content.strip()
            state, observation, feedback = parse_state_response(raw)

            if state:
                _current_state = state
                _state_log.append({
                    "state": state,
                    "observation": observation,
                    "feedback": feedback,
                })

            print(f"[proactive] [{state or '?'}] {observation}")
            if feedback:
                print(f"[proactive]  -> {feedback}")

            _broadcast({
                "state": state,
                "observation": observation,
                "feedback": feedback,
            })
        except Exception as e:
            print(f"Proactive check error: {e}")


# ---------- MJPEG helpers ----------
def _mjpeg_stream(get_frame, fps=4):
    delay = 1.0 / fps
    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        time.sleep(delay)


# ---------- Routes ----------
@app.route("/")
def index():
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>SousChef Live</title>
        <style>
          body { margin: 0; background: #111; color: #eee;
                 font-family: system-ui, -apple-system, sans-serif; }
          .wrap { display: grid; grid-template-columns: 1fr 1fr;
                  gap: 12px; padding: 12px; height: calc(100vh - 120px); box-sizing: border-box; }
          .panel { background: #1a1a1a; border-radius: 12px; padding: 10px;
                   display: flex; flex-direction: column; }
          .title { font-size: 14px; opacity: 0.85; margin-bottom: 8px; }
          img { width: 100%; height: 100%; object-fit: contain;
                border-radius: 8px; background: #000; }
          .imgwrap { flex: 1; }
          a { color: #7af; }
          .bar { padding: 8px 12px; font-size: 13px; display: flex; gap: 16px; }
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="panel">
            <div class="title">RGB (webcam)</div>
            <div class="imgwrap"><img src="/rgb.mjpg"></div>
          </div>
          <div class="panel">
            <div class="title">Thermal (MLX90640)</div>
            <div class="imgwrap">
              <img src="/thermal.mjpg" style="image-rendering: pixelated;">
            </div>
          </div>
        </div>
        <div class="bar">
          <a href="/composite">Composite view</a>
        </div>

        <!-- Voice assistant -->
        <div class="voice-bar">
          <button id="btn" title="Tap to ask">🎙️</button>
          <div class="voice-text">
            <div id="heard"></div>
            <div id="answer"></div>
            <div id="status">Tap the mic to ask a question</div>
          </div>
        </div>

        <div id="feedback" class="fb">
          <div class="fb-row">
            <span id="fb-state" class="fb-badge"></span>
            <span id="fb-obs"></span>
          </div>
          <div id="fb-tip" class="fb-tip"></div>
        </div>

        <style>
          .voice-bar {
            position: fixed; bottom: 0; left: 0; right: 0;
            background: rgba(20,20,20,0.96); border-top: 2px solid #333;
            padding: 12px 20px; display: flex; align-items: center; gap: 16px;
            z-index: 200;
          }
          #btn {
            width: 56px; height: 56px; border-radius: 50%; border: none;
            font-size: 24px; cursor: pointer; background: #2a2a2a;
            flex-shrink: 0;
            box-shadow: 0 0 0 0 rgba(200,60,60,0);
            transition: background 0.2s, box-shadow 0.2s;
          }
          #btn.listening {
            background: #b02020;
            box-shadow: 0 0 0 8px rgba(200,60,60,0.25);
            animation: pulse 1.2s ease-in-out infinite;
          }
          #btn.thinking { background: #2a4a2a; }
          @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 6px rgba(200,60,60,0.2); }
            50%       { box-shadow: 0 0 0 14px rgba(200,60,60,0.05); }
          }
          .voice-text { flex: 1; min-width: 0; }
          #heard  { font-size: 13px; color: #888; font-style: italic; min-height: 18px; }
          #answer { font-size: 15px; line-height: 1.4; min-height: 20px; margin-top: 2px; }
          #status { font-size: 12px; color: #555; margin-top: 2px; }
          .fb {
            position: fixed; bottom: -120px; left: 50%; transform: translateX(-50%);
            background: rgba(20,20,20,0.96); border-top: 3px solid #f59e0b;
            border-radius: 12px; padding: 14px 22px; max-width: 620px; width: 90%;
            font-size: 15px; line-height: 1.5; z-index: 100;
            transition: bottom 0.4s ease; box-shadow: 0 -2px 24px rgba(0,0,0,0.5);
          }
          .fb.show { bottom: 90px; }
          .fb-row { display: flex; align-items: flex-start; gap: 10px; }
          .fb-badge {
            flex-shrink: 0; background: #f59e0b; color: #111; font-size: 11px;
            font-weight: 700; padding: 3px 8px; border-radius: 4px;
            letter-spacing: 0.05em; margin-top: 2px;
          }
          .fb-tip { margin-top: 8px; color: #f59e0b; font-style: italic; }
        </style>
        <script>
          // Voice assistant
          const btn      = document.getElementById('btn');
          const heardEl  = document.getElementById('heard');
          const answerEl = document.getElementById('answer');
          const statusEl = document.getElementById('status');

          const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
          if (!SR) {
            statusEl.textContent = 'Speech recognition not supported — try Chrome or Edge.';
            btn.disabled = true;
          }

          btn.addEventListener('click', () => {
            if (!SR || btn.disabled) return;
            heardEl.textContent  = '';
            answerEl.textContent = '';
            const rec = new SR();
            rec.lang = 'en-US';
            rec.interimResults = false;
            rec.maxAlternatives = 1;

            btn.className = 'listening';
            statusEl.textContent = 'Listening…';
            rec.start();

            rec.onresult = (e) => {
              const q = e.results[0][0].transcript;
              heardEl.textContent = '\u201c' + q + '\u201d';
              askVoice(q);
            };
            rec.onerror = (e) => {
              btn.className = '';
              statusEl.textContent = 'Mic error: ' + e.error + '. Tap to retry.';
            };
            rec.onend = () => {
              if (btn.className === 'listening') {
                btn.className = 'thinking';
                statusEl.textContent = 'Thinking…';
              }
            };
          });

          async function askVoice(question) {
            btn.className = 'thinking';
            statusEl.textContent = 'Thinking…';
            try {
              const res  = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
              });
              const data = await res.json();
              answerEl.textContent = data.answer;
              btn.className = '';
              statusEl.textContent = 'Tap the mic to ask another question';
              speak(data.answer);
            } catch (err) {
              btn.className = '';
              answerEl.textContent = 'Network error — is the server running?';
              statusEl.textContent = 'Tap to retry';
            }
          }

          let speaking = false;
          const _audio = new Audio();
          _audio.addEventListener('ended', () => { speaking = false; });
          _audio.addEventListener('error', () => { speaking = false; });

          async function speak(text) {
            speaking = true;
            try {
              const res = await fetch('/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
              });
              if (!res.ok) { speaking = false; return; }
              const blob = await res.blob();
              const url = URL.createObjectURL(blob);
              _audio.src = url;
              _audio.play();
            } catch (e) {
              console.error('TTS error:', e);
              speaking = false;
            }
          }

          // SSE for proactive observations
          const _es = new EventSource('/events');
          const _fb = document.getElementById('feedback');
          let _fbT;
          _es.onmessage = (e) => {
            const d = JSON.parse(e.data);
            document.getElementById('fb-state').textContent = d.state || '';
            document.getElementById('fb-obs').textContent = d.observation || '';
            const tip = document.getElementById('fb-tip');
            tip.textContent = d.feedback || '';
            tip.style.display = d.feedback ? 'block' : 'none';
            _fb.classList.add('show');
            clearTimeout(_fbT);
            _fbT = setTimeout(() => _fb.classList.remove('show'), 14000);
            // Only speak if there's actionable feedback
            if (d.feedback && btn.className === '' && !speaking) {
              speak(d.feedback);
            }
          };
        </script>
      </body>
    </html>
    """

@app.route("/rgb.mjpg")
def rgb_mjpg():
    def get():
        with _lock: return _rgb
    return Response(_mjpeg_stream(get), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/thermal.mjpg")
def thermal_mjpg():
    def get():
        with _lock: return _thermal
    return Response(_mjpeg_stream(get), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/latest")
def latest():
    with _lock:
        frame = _composite
    if frame is None:
        return "Capture starting up, try again in a moment.", 503
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return "Encode error", 500
    return Response(jpg.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})

@app.route("/composite")
def composite_page():
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>SousChef Composite</title>
        <style>
          body { margin: 0; background: #111; display: flex;
                 flex-direction: column; align-items: center; }
          img  { max-width: 100%; border-radius: 8px; margin-top: 12px; }
          p    { color: #888; font-family: system-ui; font-size: 13px; }
        </style>
      </head>
      <body>
        <img id="c" src="/latest">
        <p id="ts">Loading...</p>
        <div id="feedback" class="fb">
          <div class="fb-row">
            <span id="fb-state" class="fb-badge"></span>
            <span id="fb-obs"></span>
          </div>
          <div id="fb-tip" class="fb-tip"></div>
        </div>

        <style>
          .fb {
            position: fixed; bottom: -120px; left: 50%; transform: translateX(-50%);
            background: rgba(20,20,20,0.96); border-top: 3px solid #f59e0b;
            border-radius: 12px; padding: 14px 22px; max-width: 620px; width: 90%;
            font-size: 15px; line-height: 1.5; color: #eee; font-family: system-ui, sans-serif;
            z-index: 100; transition: bottom 0.4s ease;
            box-shadow: 0 -2px 24px rgba(0,0,0,0.5);
          }
          .fb.show { bottom: 16px; }
          .fb-row { display: flex; align-items: flex-start; gap: 10px; }
          .fb-badge {
            flex-shrink: 0; background: #f59e0b; color: #111; font-size: 11px;
            font-weight: 700; padding: 3px 8px; border-radius: 4px;
            letter-spacing: 0.05em; margin-top: 2px;
          }
          .fb-tip { margin-top: 8px; color: #f59e0b; font-style: italic; }
        </style>
        <script>
          setInterval(() => {
            document.getElementById('c').src = '/latest?t=' + Date.now();
            document.getElementById('ts').textContent =
              'Last updated: ' + new Date().toLocaleTimeString();
          }, 500);

          const _es = new EventSource('/events');
          const _fb = document.getElementById('feedback');
          let _fbT;
          _es.onmessage = (e) => {
            const d = JSON.parse(e.data);
            document.getElementById('fb-state').textContent = d.state || '';
            document.getElementById('fb-obs').textContent = d.observation || '';
            const tip = document.getElementById('fb-tip');
            tip.textContent = d.feedback || '';
            tip.style.display = d.feedback ? 'block' : 'none';
            _fb.classList.add('show');
            clearTimeout(_fbT);
            _fbT = setTimeout(() => _fb.classList.remove('show'), 14000);
          };
        </script>
      </body>
    </html>
    """

@app.route("/events")
def events():
    """SSE stream — pushes proactive observations to the browser."""
    q = queue.Queue()
    with _sse_listeners_lock:
        _sse_listeners.append(q)

    def stream():
        try:
            while True:
                try:
                    data = q.get(timeout=20)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _sse_listeners_lock:
                if q in _sse_listeners:
                    _sse_listeners.remove(q)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/tts", methods=["POST"])
def tts():
    """Generate MP3 audio from text using edge-tts."""
    text = (request.get_json() or {}).get("text", "").strip()
    if not text:
        return "No text provided", 400
    try:
        audio = _tts_bytes(text)
        return Response(audio, mimetype="audio/mpeg")
    except Exception as e:
        print(f"TTS error: {e}")
        return "TTS generation failed", 500


@app.route("/ask", methods=["POST"])
def ask():
    """Receive a question, send current frame + thermal data to Claude, return answer."""
    question = (request.get_json() or {}).get("question", "").strip()
    if not question:
        return jsonify({"answer": "I didn't catch that — could you try again?"})

    with _lock:
        frame   = _rgb.copy() if _rgb is not None else None
        t_min_v = _t_min
        t_max_v = _t_max
        t_avg_v = _t_avg

    if frame is None:
        return jsonify({"answer": "Camera isn't ready yet. Give me a moment."})

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return jsonify({"answer": "Couldn't grab the camera frame."})

    img_b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")

    try:
        msg = ai.chat.completions.create(
            model="anthropic/claude-opus-4.6",
            max_tokens=80,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SousChef, a cooking assistant with a kitchen camera and thermal sensor. "
                        "Answer in 1 sentence — 2 max if essential. Be direct, like a chef calling out instructions."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Thermal sensor — min: {t_min_v:.1f}°C, "
                                f"max: {t_max_v:.1f}°C, avg: {t_avg_v:.1f}°C.\n"
                                f"Question: {question}"
                            )
                        }
                    ]
                }
            ]
        )
        answer = msg.choices[0].message.content
    except Exception as e:
        print(f"Claude API error: {e}")
        answer = "Sorry, I had trouble thinking that through. Try again?"

    return jsonify({"answer": answer})


@app.route("/voice")
def voice_page():
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>SousChef Voice</title>
        <style>
          * { box-sizing: border-box; }
          body {
            margin: 0; background: #111; color: #eee;
            font-family: system-ui, -apple-system, sans-serif;
            display: flex; flex-direction: column; align-items: center;
            padding: 48px 24px;
          }
          h1 { font-size: 22px; margin-bottom: 36px; letter-spacing: 0.05em; }
          #btn {
            width: 110px; height: 110px; border-radius: 50%; border: none;
            font-size: 42px; cursor: pointer; background: #2a2a2a;
            box-shadow: 0 0 0 0 rgba(200,60,60,0);
            transition: background 0.2s, box-shadow 0.2s;
          }
          #btn.listening {
            background: #b02020;
            box-shadow: 0 0 0 12px rgba(200,60,60,0.25);
            animation: pulse 1.2s ease-in-out infinite;
          }
          #btn.thinking { background: #2a4a2a; }
          @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 8px rgba(200,60,60,0.2); }
            50%       { box-shadow: 0 0 0 20px rgba(200,60,60,0.05); }
          }
          #heard   { margin-top: 28px; font-size: 15px; color: #888; min-height: 22px; font-style: italic; }
          #answer  { margin-top: 16px; font-size: 19px; line-height: 1.55;
                     max-width: 480px; text-align: center; min-height: 28px; }
          #status  { margin-top: 20px; font-size: 13px; color: #555; }
          a        { color: #7af; font-size: 13px; margin-top: 36px; }
        </style>
      </head>
      <body>
        <h1>SousChef</h1>
        <button id="btn" title="Tap to ask">🎙️</button>
        <div id="heard"></div>
        <div id="answer"></div>
        <div id="status">Tap the mic to ask a question</div>
        <a href="/">Back to live view</a>

        <script>
          const btn      = document.getElementById('btn');
          const heardEl  = document.getElementById('heard');
          const answerEl = document.getElementById('answer');
          const statusEl = document.getElementById('status');

          const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
          if (!SR) {
            statusEl.textContent = 'Speech recognition not supported — try Chrome or Edge.';
            btn.disabled = true;
          }

          let speaking = false;
          const _audio = new Audio();
          _audio.addEventListener('ended', () => { speaking = false; });
          _audio.addEventListener('error', () => { speaking = false; });

          async function speak(text) {
            speaking = true;
            try {
              const res = await fetch('/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
              });
              if (!res.ok) { speaking = false; return; }
              const blob = await res.blob();
              const url = URL.createObjectURL(blob);
              _audio.src = url;
              _audio.play();
            } catch (e) {
              console.error('TTS error:', e);
              speaking = false;
            }
          }

          btn.addEventListener('click', () => {
            if (!SR || btn.disabled) return;
            heardEl.textContent  = '';
            answerEl.textContent = '';

            const rec = new SR();
            rec.lang = 'en-US';
            rec.interimResults = false;
            rec.maxAlternatives = 1;

            btn.className = 'listening';
            statusEl.textContent = 'Listening…';
            rec.start();

            rec.onresult = (e) => {
              const q = e.results[0][0].transcript;
              heardEl.textContent = '\u201c' + q + '\u201d';
              ask(q);
            };

            rec.onerror = (e) => {
              btn.className = '';
              statusEl.textContent = 'Mic error: ' + e.error + '. Tap to retry.';
            };

            rec.onend = () => {
              if (btn.className === 'listening') {
                btn.className = 'thinking';
                statusEl.textContent = 'Thinking…';
              }
            };
          });

          async function ask(question) {
            btn.className = 'thinking';
            statusEl.textContent = 'Thinking…';
            try {
              const res  = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
              });
              const data = await res.json();
              answerEl.textContent = data.answer;
              btn.className = '';
              statusEl.textContent = 'Listening for cooking activity…';
              speak(data.answer);
            } catch (err) {
              btn.className = '';
              answerEl.textContent = 'Network error — is the server running?';
              statusEl.textContent = 'Tap to retry';
            }
          }

          // Proactive observations via SSE
          const es = new EventSource('/events');
          es.onmessage = (e) => {
            const d = JSON.parse(e.data);
            // Don't interrupt if user is mid-question or already speaking
            if (btn.className !== '' || speaking) return;
            let text = d.observation || '';
            if (d.feedback) text += ' ' + d.feedback;
            answerEl.textContent = text;
            heardEl.textContent = d.state ? '[' + d.state + ']' : '';
            // Only speak if there's actionable feedback
            if (d.feedback) speak(d.feedback);
          };
          es.onerror = () => { /* reconnects automatically */ };
        </script>
      </body>
    </html>
    """


# ---------- Start ----------
if __name__ == "__main__":
    if "--mock" in sys.argv:
        video = os.path.join(os.path.dirname(__file__), "..", "data", "test.mp4")
        for i, arg in enumerate(sys.argv):
            if arg == "--video" and i + 1 < len(sys.argv):
                video = sys.argv[i + 1]
        threading.Thread(target=mock_capture_loop, args=(video,), daemon=True).start()
    else:
        threading.Thread(target=capture_loop, daemon=True).start()

    threading.Thread(target=proactive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=8000, threaded=True)

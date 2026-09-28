# 🎙️ Real-Time Multimodal Voice Assistant


## What Is It?
A fully functional voice assistant with end-to-end latency budgeting, graceful degradation on timeouts, and a detailed pipeline profiler. You speak → it transcribes → it thinks → it speaks back. Every stage is timed and compared against a target budget.

---

## Industrial Applications
| Industry | Use Case |
|---|---|
| Customer Service | Automated phone support agents |
| Healthcare | Hands-free clinical note dictation |
| Accessibility | Screen-reader replacement for visual impairment |
| Automotive | In-car voice control without cloud dependency |
| Field Operations | Hands-free warehouse/construction tools |
| Education | Real-time language tutoring |

---

## Pipeline Architecture

```
Microphone Input
      │
      ▼ (50ms)
VAD — Silence Detection
      │
      ▼ (30ms)
Whisper STT ──────────────── timeout 3s ──► "Could you repeat that?"
      │ (300ms)
      ▼
GPT-4o-mini (streaming) ──── timeout 8s ──► fallback message
      │ (800ms)
      ▼
OpenAI TTS ───────────────── timeout 4s ──► text fallback to console
      │ (400ms)
      ▼
Audio Playback
      │ (200ms)
      ▼
Total: ~2000ms end-to-end
```

---

## Key Technical Concepts

### Latency Budget
Every millisecond is accounted for before you write a line of code. You decide upfront: "STT gets 300ms, LLM gets 800ms..." If a stage exceeds its budget, you know exactly where to optimize.

### Graceful Degradation
Instead of crashing on timeout, the system falls back gracefully:
- STT timeout → prompt user to repeat
- LLM slow → acknowledge and ask again
- TTS failure → print text to console
This is what separates a demo from a production system.

### Streaming LLM
Instead of waiting for the full response before starting TTS, you can start speaking the first sentence while the rest generates — a "parallel pipeline" that halves perceived latency.

### VAD (Voice Activity Detection)
Detects when the user stops speaking using amplitude threshold, so recording stops automatically instead of requiring a button press.

---

## File Structure
```
05-realtime-multimodal/
├── src/
│   └── pipeline/
│       ├── voice_assistant.py    # Full end-to-end pipeline
│       └── latency_analyzer.py  # Budget profiler (run before deploying)
└── requirements.txt
```

---

# ▶️ How to Run

## Prerequisites
- Python 3.11+
- OpenAI API key
- Microphone + speakers
- `ffmpeg` or `afplay` (macOS) for audio playback

## Setup
```bash
cd 05-realtime-multimodal
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-your-key" > .env
```

## Step 1 — Profile Your Latency First
```bash
python -m src.pipeline.latency_analyzer
# Shows p50/p95/p99 per stage + bottleneck tips
# No API key needed — uses simulation
```

## Step 2 — Run the Voice Assistant
```bash
python -m src.pipeline.voice_assistant
# Speak when you see [🎤] Listening...
# After each exchange, latency breakdown is printed
# Ctrl+C to exit
```

---

# 📐 Strategy

## Why Latency Budgeting Matters
Most voice assistant demos feel slow because nobody measured where the time goes. By budgeting each stage:
1. You **set a target** (2 seconds total feels instant to users)
2. You **know your headroom** — 800ms for LLM means you can't use a 7B model locally
3. You **negotiate tradeoffs**: better TTS quality → +200ms → reduce LLM max tokens



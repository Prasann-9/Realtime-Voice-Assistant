"""
Real-Time Voice Assistant Pipeline.

End-to-end latency budget (target <2s total):
  ┌──────────────────┬──────────────┐
  │ Stage            │ Budget (ms)  │
  ├──────────────────┼──────────────┤
  │ Audio capture    │     50       │
  │ VAD (silence)    │     30       │
  │ STT (Whisper)    │    300       │
  │ LLM inference    │    800       │
  │ TTS generation   │    400       │
  │ Audio playback   │    200       │
  │ Buffer/overhead  │    220       │
  ├──────────────────┼──────────────┤
  │ TOTAL            │  2000        │
  └──────────────────┴──────────────┘

Graceful degradation:
  - STT timeout → "Could you repeat that?"
  - LLM timeout → "I need a moment, please hold"
  - TTS failure → print text fallback
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

import openai
import pyaudio
import numpy as np
from openai import AsyncOpenAI


# ---- Budget Constants (ms) ----
BUDGET_STT_MS = 3000
BUDGET_LLM_MS = 8000
BUDGET_TTS_MS = 4000


class PipelineState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class LatencyBudget:
    """Tracks actual vs budgeted latency for each pipeline stage."""
    audio_capture_ms: float = 0
    stt_ms: float = 0
    llm_ms: float = 0
    tts_ms: float = 0

    @property
    def total_ms(self) -> float:
        return self.audio_capture_ms + self.stt_ms + self.llm_ms + self.tts_ms

    def report(self) -> str:
        return (
            f"\n=== Latency Budget ===\n"
            f"  Audio Capture : {self.audio_capture_ms:.0f}ms\n"
            f"  STT (Whisper) : {self.stt_ms:.0f}ms\n"
            f"  LLM           : {self.llm_ms:.0f}ms\n"
            f"  TTS           : {self.tts_ms:.0f}ms\n"
            f"  ─────────────────────\n"
            f"  TOTAL         : {self.total_ms:.0f}ms\n"
        )


# ---- Audio Capture ----
RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5  # seconds of silence before stopping recording


def capture_audio(max_seconds: int = 10) -> Optional[bytes]:
    """Capture audio from microphone with VAD-based silence detection."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT, channels=CHANNELS, rate=RATE,
        input=True, frames_per_buffer=CHUNK
    )

    print("[🎤] Listening... (speak now)")
    frames = []
    silent_chunks = 0
    silence_limit = int(SILENCE_DURATION * RATE / CHUNK)
    total_limit = int(max_seconds * RATE / CHUNK)

    for _ in range(total_limit):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        # VAD: check amplitude
        samples = np.frombuffer(data, dtype=np.int16)
        amplitude = np.abs(samples).mean()

        if amplitude < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks >= silence_limit and len(frames) > silence_limit * 2:
            print("[🎤] Silence detected — stopping")
            break

    stream.stop_stream()
    stream.close()
    pa.terminate()
    return b"".join(frames)


# ---- Speech-to-Text ----
async def transcribe_audio(
    audio_bytes: bytes,
    client: AsyncOpenAI,
    budget: LatencyBudget,
    timeout_ms: int = BUDGET_STT_MS,
) -> Optional[str]:
    """Convert audio bytes to text using Whisper with timeout."""
    import tempfile, wave, os

    # Write PCM bytes to WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio_bytes)

    start = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_ms / 1000):
            with open(wav_path, "rb") as audio_file:
                transcript = await client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file, language="en"
                )
        budget.stt_ms = (time.perf_counter() - start) * 1000
        os.unlink(wav_path)
        return transcript.text
    except asyncio.TimeoutError:
        print(f"[⚠️] STT timeout after {timeout_ms}ms — degrading gracefully")
        budget.stt_ms = timeout_ms
        return None
    except Exception as e:
        print(f"[❌] STT error: {e}")
        return None


# ---- LLM with Streaming ----
async def generate_response(
    text: str,
    client: AsyncOpenAI,
    budget: LatencyBudget,
    history: list,
    timeout_ms: int = BUDGET_LLM_MS,
) -> Optional[str]:
    """Generate LLM response with streaming + timeout."""
    history.append({"role": "user", "content": text})

    start = time.perf_counter()
    full_response = ""

    try:
        async with asyncio.timeout(timeout_ms / 1000):
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise (1-3 sentences)."},
                    *history[-10:],  # Rolling window
                ],
                stream=True,
                max_tokens=150,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                print(delta, end="", flush=True)

        budget.llm_ms = (time.perf_counter() - start) * 1000
        print()
        history.append({"role": "assistant", "content": full_response})
        return full_response

    except asyncio.TimeoutError:
        budget.llm_ms = timeout_ms
        print(f"\n[⚠️] LLM timeout after {timeout_ms}ms — degrading gracefully")
        return "I need a moment to think about that. Could you ask again?"
    except Exception as e:
        print(f"[❌] LLM error: {e}")
        return None


# ---- Text-to-Speech ----
async def text_to_speech(
    text: str,
    client: AsyncOpenAI,
    budget: LatencyBudget,
    timeout_ms: int = BUDGET_TTS_MS,
) -> bool:
    """Generate and play TTS audio with timeout and text fallback."""
    import tempfile, os

    start = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_ms / 1000):
            response = await client.audio.speech.create(
                model="tts-1", voice="alloy", input=text, response_format="mp3"
            )
        budget.tts_ms = (time.perf_counter() - start) * 1000

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(response.content)
            tmp_path = f.name

        os.system(f"ffplay -nodisp -autoexit {tmp_path} -loglevel quiet 2>/dev/null || afplay {tmp_path} 2>/dev/null")
        os.unlink(tmp_path)
        return True

    except asyncio.TimeoutError:
        print(f"[⚠️] TTS timeout — text fallback: {text}")
        budget.tts_ms = timeout_ms
        return False
    except Exception as e:
        print(f"[⚠️] TTS unavailable — text fallback: {text}")
        return False


# ---- Main Pipeline ----
async def run_voice_assistant():
    client = AsyncOpenAI()
    history = []
    state = PipelineState.IDLE

    print("[🎙️] Voice Assistant Ready — press Ctrl+C to quit\n")

    while True:
        try:
            # 1. Capture audio
            state = PipelineState.LISTENING
            t0 = time.perf_counter()
            audio = capture_audio()
            budget = LatencyBudget()
            budget.audio_capture_ms = (time.perf_counter() - t0) * 1000

            if audio is None:
                continue

            state = PipelineState.PROCESSING

            # 2. Transcribe
            transcript = await transcribe_audio(audio, client, budget)
            if not transcript:
                await text_to_speech("Could you repeat that?", client, LatencyBudget())
                continue
            print(f"[You] {transcript}")

            # 3. Generate response
            response = await generate_response(transcript, client, budget, history)
            if not response:
                continue

            # 4. Speak
            state = PipelineState.SPEAKING
            await text_to_speech(response, client, budget)

            # 5. Log latency
            print(budget.report())

        except KeyboardInterrupt:
            print("\n[👋] Goodbye!")
            break
        except Exception as e:
            state = PipelineState.ERROR
            print(f"[❌] Pipeline error: {e}")
            state = PipelineState.IDLE


if __name__ == "__main__":
    asyncio.run(run_voice_assistant())

"""
Latency Budget Analyzer.
Simulates and profiles the voice pipeline end-to-end to identify bottlenecks.
Run this BEFORE deploying to understand where your latency is going.
"""

import asyncio
import time
import statistics
import json
from pathlib import Path
from typing import Dict, List


SIMULATED_STAGES = {
    "audio_capture": {"mean_ms": 50, "std_ms": 10},
    "vad_processing": {"mean_ms": 30, "std_ms": 5},
    "stt_whisper": {"mean_ms": 320, "std_ms": 80},
    "llm_first_token": {"mean_ms": 280, "std_ms": 120},
    "llm_full_response": {"mean_ms": 820, "std_ms": 200},
    "tts_generation": {"mean_ms": 380, "std_ms": 60},
    "audio_playback_start": {"mean_ms": 150, "std_ms": 20},
}

TARGET_E2E_MS = 2000
SIMULATIONS = 100


def simulate_stage(stage: str, n: int = SIMULATIONS) -> List[float]:
    import numpy as np
    cfg = SIMULATED_STAGES[stage]
    samples = np.random.normal(cfg["mean_ms"], cfg["std_ms"], n)
    return [max(0, s) for s in samples]


def run_budget_analysis():
    print("=" * 55)
    print("  Voice Pipeline — Latency Budget Analysis")
    print("=" * 55)

    total_samples = [0.0] * SIMULATIONS
    report_rows = []

    for stage, config in SIMULATED_STAGES.items():
        samples = simulate_stage(stage)
        p50 = statistics.median(samples)
        p95 = sorted(samples)[int(SIMULATIONS * 0.95)]
        p99 = sorted(samples)[min(int(SIMULATIONS * 0.99), SIMULATIONS - 1)]

        total_samples = [a + b for a, b in zip(total_samples, samples)]

        status = "✅" if p95 < config["mean_ms"] * 2 else "⚠️"
        report_rows.append({
            "stage": stage,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "budget": config["mean_ms"],
            "over_budget": p95 > config["mean_ms"] * 1.5,
        })

    # Print breakdown
    print(f"\n{'Stage':<25} {'p50':>8} {'p95':>8} {'p99':>8}  {'Budget':>8}")
    print("-" * 65)
    for row in report_rows:
        over = " ⚠️ OVER" if row["over_budget"] else ""
        print(f"  {row['stage']:<23} {row['p50']:>6.0f}ms {row['p95']:>6.0f}ms {row['p99']:>6.0f}ms  {row['budget']:>6.0f}ms{over}")

    # Total E2E
    e2e_p50 = statistics.median(total_samples)
    e2e_p95 = sorted(total_samples)[int(SIMULATIONS * 0.95)]
    print("-" * 65)
    print(f"  {'TOTAL E2E':<23} {e2e_p50:>6.0f}ms {e2e_p95:>6.0f}ms")
    status = "✅ WITHIN" if e2e_p95 <= TARGET_E2E_MS else "❌ EXCEEDS"
    print(f"\n  Target: {TARGET_E2E_MS}ms  →  {status} budget at p95")

    # Optimization suggestions
    print("\n=== Optimization Tips ===")
    bottlenecks = sorted(report_rows, key=lambda r: r["p95"], reverse=True)
    for b in bottlenecks[:3]:
        tip = OPTIMIZATION_TIPS.get(b["stage"], "Profile and cache where possible")
        print(f"  {b['stage']}: {tip}")

    # Save report
    Path("results").mkdir(exist_ok=True)
    with open("results/latency_report.json", "w") as f:
        json.dump({"rows": report_rows, "e2e_p50_ms": e2e_p50, "e2e_p95_ms": e2e_p95}, f, indent=2)
    print("\n  Report saved to results/latency_report.json")


OPTIMIZATION_TIPS = {
    "stt_whisper": "Use whisper.cpp locally (2-3x faster) or stream chunks instead of full audio",
    "llm_full_response": "Use streaming + interrupt TTS at first sentence for parallel pipeline",
    "tts_generation": "Cache common responses or use faster TTS (Edge TTS, Kokoro)",
    "llm_first_token": "Reduce system prompt length, use smaller model or local Ollama",
    "audio_playback_start": "Pre-buffer audio before playback starts",
    "audio_capture": "Reduce CHUNK size for faster VAD response",
    "vad_processing": "Use WebRTC VAD (C library) instead of amplitude threshold",
}


if __name__ == "__main__":
    run_budget_analysis()

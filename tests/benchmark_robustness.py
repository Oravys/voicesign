# VoiceSign by Oravys Inc. (https://oravys.com)
#
# Standalone robustness benchmark for VoiceSign watermarking.
# Measures watermark survival across lossy codecs and signal transforms.
#
# Usage:
#   python tests/benchmark_robustness.py
#
"""
Generates a synthetic test signal, embeds a VoiceSign watermark, then
applies a battery of transformations and reports detection results in a
formatted table.
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

# Ensure the parent package is importable when running as a script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import voicesign
from voicesign.core import DETECTION_THRESHOLD

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IDENTITY = "test-user"
SALT = "test-salt"
SAMPLE_RATE = 16000
DURATION = 3.0

HAS_FFMPEG: bool = shutil.which("ffmpeg") is not None

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def generate_test_wav(
    duration: float = DURATION,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Generate a 16-bit mono WAV with a sine sweep + noise (speech-like)."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Logarithmic sine sweep 200 Hz -> 4000 Hz
    f0, f1 = 200.0, 4000.0
    sweep = 0.4 * np.sin(
        2 * np.pi * f0 * duration
        / np.log(f1 / f0)
        * (np.exp(t / duration * np.log(f1 / f0)) - 1)
    )

    # Add pink-ish noise
    rng = np.random.RandomState(12345)
    white = rng.randn(n_samples)
    pink = np.cumsum(white)
    pink = pink - np.mean(pink)
    pink = pink / (np.max(np.abs(pink)) + 1e-12) * 0.15

    samples_f = np.clip(sweep + pink, -1.0, 1.0)
    int_samples = (samples_f * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


def wav_to_samples(wav_bytes: bytes) -> tuple:
    """Return (float64 samples, sample_rate) from WAV bytes."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return arr, sr


def samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float64 samples to 16-bit mono WAV."""
    clipped = np.clip(samples, -1.0, 1.0)
    int_samples = (clipped * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


def ffmpeg_transcode(wav_bytes: bytes, output_ext: str, extra_args: list) -> bytes:
    """Transcode WAV -> lossy format -> WAV via ffmpeg. Returns WAV bytes."""
    # Use directory next to this script (not %TEMP%) because ffmpeg may be
    # blocked from writing to the system temp dir on some Windows setups.
    _bench_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = tempfile.mkdtemp(prefix=".voicesign_bench_", dir=_bench_dir)
    path_in = os.path.join(tmp_dir, "input.wav")
    path_mid = os.path.join(tmp_dir, "mid.{}".format(output_ext))
    path_out = os.path.join(tmp_dir, "output.wav")
    try:
        with open(path_in, "wb") as f:
            f.write(wav_bytes)

        cmd_encode = ["ffmpeg", "-y", "-i", path_in, *extra_args, path_mid]
        result = subprocess.run(
            cmd_encode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[:300])

        cmd_decode = [
            "ffmpeg", "-y", "-i", path_mid,
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16",
            "-f", "wav", path_out,
        ]
        result = subprocess.run(
            cmd_decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[:300])

        with open(path_out, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark():
    print()
    print("VoiceSign Robustness Benchmark")
    print("=" * 78)
    print()

    # --- Generate and sign ---
    print("Generating test audio ({:.1f}s, {} Hz)...".format(DURATION, SAMPLE_RATE))
    raw_wav = generate_test_wav()

    print("Signing with identity='{}', salt='{}'...".format(IDENTITY, SALT))
    signed_wav = voicesign.sign(raw_wav, identity=IDENTITY, salt=SALT)
    print("Signed WAV size: {} bytes".format(len(signed_wav)))
    print()

    if not HAS_FFMPEG:
        print("[WARNING] ffmpeg not found -- codec tests will be skipped.")
        print()

    # --- Define transformations ---
    results = []  # list of (label, result_dict)

    # Baseline
    res = voicesign.verify(signed_wav, identity=IDENTITY, salt=SALT)
    results.append(("Baseline (no transform)", res))

    # FFmpeg codec transforms
    codec_transforms = [
        ("MP3 64 kbps", "mp3", ["-b:a", "64k"]),
        ("MP3 128 kbps", "mp3", ["-b:a", "128k"]),
        ("MP3 192 kbps", "mp3", ["-b:a", "192k"]),
        ("MP3 320 kbps", "mp3", ["-b:a", "320k"]),
        ("Opus 64 kbps", "ogg", ["-c:a", "libopus", "-b:a", "64k"]),
        ("AAC 128 kbps", "m4a", ["-c:a", "aac", "-b:a", "128k"]),
        ("WAV lossless roundtrip", "wav", ["-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16"]),
    ]

    for label, ext, args in codec_transforms:
        if not HAS_FFMPEG:
            results.append((label, None))
            continue
        try:
            transformed = ffmpeg_transcode(signed_wav, ext, args)
            res = voicesign.verify(transformed, identity=IDENTITY, salt=SALT)
            results.append((label, res))
        except RuntimeError as exc:
            results.append((label, None))

    # Signal-level transforms (numpy, no ffmpeg)
    samples, sr = wav_to_samples(signed_wav)

    # White noise SNR 30 dB
    rng = np.random.RandomState(42)
    sig_pow = np.mean(samples ** 2)
    noise_pow_30 = sig_pow / (10 ** (30 / 10))
    noisy_30 = samples + rng.randn(len(samples)) * np.sqrt(noise_pow_30)
    results.append(("White noise SNR 30 dB",
                     voicesign.verify(samples_to_wav(noisy_30, sr), identity=IDENTITY, salt=SALT)))

    # White noise SNR 20 dB
    rng = np.random.RandomState(42)
    noise_pow_20 = sig_pow / (10 ** (20 / 10))
    noisy_20 = samples + rng.randn(len(samples)) * np.sqrt(noise_pow_20)
    results.append(("White noise SNR 20 dB",
                     voicesign.verify(samples_to_wav(noisy_20, sr), identity=IDENTITY, salt=SALT)))

    # Volume 0.5x
    results.append(("Volume 0.5x",
                     voicesign.verify(samples_to_wav(samples * 0.5, sr), identity=IDENTITY, salt=SALT)))

    # Volume 2.0x (clipped)
    results.append(("Volume 2.0x (clipped)",
                     voicesign.verify(samples_to_wav(samples * 2.0, sr), identity=IDENTITY, salt=SALT)))

    # Trim first 0.5s
    trim_n = int(0.5 * sr)
    results.append(("Trim first 0.5s",
                     voicesign.verify(samples_to_wav(samples[trim_n:], sr), identity=IDENTITY, salt=SALT)))

    # Trim last 0.5s
    results.append(("Trim last 0.5s",
                     voicesign.verify(samples_to_wav(samples[:-trim_n], sr), identity=IDENTITY, salt=SALT)))

    # Prepend 0.3s silence
    silence = np.zeros(int(0.3 * sr))
    padded = np.concatenate([silence, samples])
    results.append(("Prepend 0.3s silence",
                     voicesign.verify(samples_to_wav(padded, sr), identity=IDENTITY, salt=SALT)))

    # Resample 16k -> 8k -> 16k (telephone simulation)
    down = samples[::2]
    x_down = np.arange(len(down))
    x_up = np.linspace(0, len(down) - 1, len(samples))
    up = np.interp(x_up, x_down, down)
    results.append(("Resample 16k->8k->16k",
                     voicesign.verify(samples_to_wav(up, sr), identity=IDENTITY, salt=SALT)))

    # --- Print results table ---
    print("-" * 78)
    hdr = "{:<35s} | {:>8s} | {:>11s} | {:>10s}".format(
        "Transformation", "Detected", "Correlation", "Confidence",
    )
    print(hdr)
    print("-" * 35 + "-+-" + "-" * 8 + "-+-" + "-" * 11 + "-+-" + "-" * 10)

    survived = 0
    tested = 0

    for label, res in results:
        if res is None:
            status = "SKIP"
            corr_str = "N/A"
            conf_str = "N/A"
        else:
            tested += 1
            det = res["detected"]
            if det:
                survived += 1
            status = "YES" if det else "NO"
            corr_str = "{:.6f}".format(res["correlation"])
            conf_str = "{:.1%}".format(res["confidence"])
        print("{:<35s} | {:>8s} | {:>11s} | {:>10s}".format(label, status, corr_str, conf_str))

    print("-" * 35 + "-+-" + "-" * 8 + "-+-" + "-" * 11 + "-+-" + "-" * 10)
    print()
    print("Survived: {}/{} tested".format(survived, tested))
    print("Detection threshold: {}".format(DETECTION_THRESHOLD))
    print()

    # --- Exit code ---
    # Return 0 even if some watermarks do not survive -- this is a measurement tool.
    return 0


if __name__ == "__main__":
    sys.exit(run_benchmark())

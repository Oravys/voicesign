# VoiceSign by Oravys Inc. (https://oravys.com)
#
# Codec survival tests -- measures how well the VoiceSign spread-spectrum
# DCT watermark survives real-world audio transformations (lossy codecs,
# noise, resampling, trimming, volume changes).
#
"""
Pytest suite that signs a synthetic WAV, applies various transformations,
and records whether the watermark is still detectable plus the correlation
and confidence values.

Usage::

    pytest tests/test_codec_survival.py -v -s

The ``-s`` flag is recommended so the printed correlation values are visible.
"""

import io
import os
import shutil
import struct
import subprocess
import tempfile
import wave

import numpy as np
import pytest

import voicesign
from voicesign.core import DETECTION_THRESHOLD

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IDENTITY = "test-user"
SALT = "test-salt"
SAMPLE_RATE = 16000
DURATION = 3.0

HAS_FFMPEG: bool = shutil.which("ffmpeg") is not None

skip_no_ffmpeg = pytest.mark.skipif(
    not HAS_FFMPEG,
    reason="ffmpeg not found on PATH -- skipping codec test",
)


def _generate_test_wav(
    duration: float = DURATION,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Generate a 16-bit mono WAV with a sine sweep + noise (speech-like)."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Logarithmic sine sweep from 200 Hz to 4000 Hz
    f0, f1 = 200.0, 4000.0
    sweep = 0.4 * np.sin(
        2 * np.pi * f0 * duration
        / np.log(f1 / f0)
        * (np.exp(t / duration * np.log(f1 / f0)) - 1)
    )

    # Add pink-ish noise to simulate speech spectral shape
    rng = np.random.RandomState(12345)
    white = rng.randn(n_samples)
    # Simple 1/f approximation via cumulative averaging
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


def _sign_wav(wav_bytes: bytes) -> bytes:
    """Sign a WAV with the default test identity/salt."""
    return voicesign.sign(wav_bytes, identity=IDENTITY, salt=SALT)


def _verify_wav(wav_bytes: bytes) -> dict:
    """Verify a WAV against the default test identity/salt."""
    return voicesign.verify(wav_bytes, identity=IDENTITY, salt=SALT)


def _wav_to_samples(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Read WAV bytes into float64 samples and return (samples, sample_rate)."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        sw = wf.getsampwidth()
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return arr, sr


def _samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
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


def _ffmpeg_transcode(wav_bytes: bytes, output_ext: str, extra_args: list[str]) -> bytes:
    """
    Transcode WAV through ffmpeg to a lossy format and back to WAV.

    Returns the round-tripped WAV bytes.
    """
    # Use a subdirectory next to this test file rather than the system temp
    # directory.  On some Windows setups ffmpeg cannot write to %TEMP% due to
    # security restrictions, whereas the project tree is always writable.
    _tests_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = tempfile.mkdtemp(prefix=".voicesign_test_", dir=_tests_dir)
    path_in = os.path.join(tmp_dir, "input.wav")
    path_mid = os.path.join(tmp_dir, f"mid.{output_ext}")
    path_out = os.path.join(tmp_dir, "output.wav")
    try:
        # Write input WAV
        with open(path_in, "wb") as f:
            f.write(wav_bytes)

        # Encode to lossy format
        cmd_encode = [
            "ffmpeg", "-y", "-i", path_in,
            *extra_args,
            path_mid,
        ]
        result = subprocess.run(
            cmd_encode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encode failed: {result.stderr.decode('utf-8', errors='replace')[:300]}"
            )

        # Decode back to WAV (16kHz mono 16-bit)
        cmd_decode = [
            "ffmpeg", "-y", "-i", path_mid,
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16",
            "-f", "wav", path_out,
        ]
        result = subprocess.run(
            cmd_decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg decode failed: {result.stderr.decode('utf-8', errors='replace')[:300]}"
            )

        with open(path_out, "rb") as f:
            return f.read()
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmp_dir, ignore_errors=True)


def _print_result(label: str, result: dict):
    """Print a single result line for visibility in pytest -s output."""
    status = "YES" if result["detected"] else "NO "
    corr = result["correlation"]
    conf = result["confidence"]
    print(f"  {label:<35s} | {status} | corr={corr:.6f} | conf={conf:.2%}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def signed_wav() -> bytes:
    """Module-scoped signed WAV -- generated once, reused across all tests."""
    raw = _generate_test_wav()
    return _sign_wav(raw)


# ---------------------------------------------------------------------------
# Baseline test
# ---------------------------------------------------------------------------

class TestBaseline:
    """Verify that the watermark is detected in the original signed WAV."""

    def test_baseline_detection(self, signed_wav):
        result = _verify_wav(signed_wav)
        _print_result("Baseline (no transform)", result)
        assert result["detected"] is True, "Baseline must always detect the watermark"
        assert result["correlation"] > DETECTION_THRESHOLD


# ---------------------------------------------------------------------------
# FFmpeg codec tests
# ---------------------------------------------------------------------------

class TestCodecSurvival:
    """Watermark survival through lossy audio codecs (requires ffmpeg)."""

    def _transcode_or_skip(self, signed_wav, label, ext, args):
        """Helper: transcode and verify, skip on ffmpeg failure."""
        try:
            transformed = _ffmpeg_transcode(signed_wav, ext, args)
        except RuntimeError as exc:
            pytest.skip(f"ffmpeg codec unavailable for {label}: {exc}")
            return None
        result = _verify_wav(transformed)
        _print_result(label, result)
        return result

    @skip_no_ffmpeg
    def test_mp3_064kbps(self, signed_wav):
        self._transcode_or_skip(signed_wav, "MP3 64 kbps", "mp3", ["-b:a", "64k"])

    @skip_no_ffmpeg
    def test_mp3_128kbps(self, signed_wav):
        self._transcode_or_skip(signed_wav, "MP3 128 kbps", "mp3", ["-b:a", "128k"])

    @skip_no_ffmpeg
    def test_mp3_192kbps(self, signed_wav):
        self._transcode_or_skip(signed_wav, "MP3 192 kbps", "mp3", ["-b:a", "192k"])

    @skip_no_ffmpeg
    def test_mp3_320kbps(self, signed_wav):
        self._transcode_or_skip(signed_wav, "MP3 320 kbps", "mp3", ["-b:a", "320k"])

    @skip_no_ffmpeg
    def test_opus_064kbps(self, signed_wav):
        self._transcode_or_skip(signed_wav, "Opus 64 kbps", "ogg", ["-c:a", "libopus", "-b:a", "64k"])

    @skip_no_ffmpeg
    def test_aac_128kbps(self, signed_wav):
        self._transcode_or_skip(signed_wav, "AAC 128 kbps", "m4a", ["-c:a", "aac", "-b:a", "128k"])

    @skip_no_ffmpeg
    def test_wav_lossless_roundtrip(self, signed_wav):
        """Lossless WAV re-encode -- must always pass."""
        result = self._transcode_or_skip(
            signed_wav, "WAV lossless roundtrip", "wav",
            ["-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16"],
        )
        if result is not None:
            assert result["detected"] is True, "Lossless roundtrip must preserve watermark"


# ---------------------------------------------------------------------------
# Numpy-based transformation tests (no ffmpeg required)
# ---------------------------------------------------------------------------

class TestSignalTransformations:
    """Watermark survival through signal-level transformations."""

    def test_white_noise_snr30(self, signed_wav):
        samples, sr = _wav_to_samples(signed_wav)
        rng = np.random.RandomState(42)
        signal_power = np.mean(samples ** 2)
        snr_linear = 10 ** (30 / 10)
        noise_power = signal_power / snr_linear
        noise = rng.randn(len(samples)) * np.sqrt(noise_power)
        noisy = samples + noise
        wav_out = _samples_to_wav(noisy, sr)
        result = _verify_wav(wav_out)
        _print_result("White noise SNR 30 dB", result)

    def test_white_noise_snr20(self, signed_wav):
        samples, sr = _wav_to_samples(signed_wav)
        rng = np.random.RandomState(42)
        signal_power = np.mean(samples ** 2)
        snr_linear = 10 ** (20 / 10)
        noise_power = signal_power / snr_linear
        noise = rng.randn(len(samples)) * np.sqrt(noise_power)
        noisy = samples + noise
        wav_out = _samples_to_wav(noisy, sr)
        result = _verify_wav(wav_out)
        _print_result("White noise SNR 20 dB", result)

    def test_volume_scale_half(self, signed_wav):
        samples, sr = _wav_to_samples(signed_wav)
        scaled = samples * 0.5
        wav_out = _samples_to_wav(scaled, sr)
        result = _verify_wav(wav_out)
        _print_result("Volume 0.5x", result)

    def test_volume_scale_double_clip(self, signed_wav):
        samples, sr = _wav_to_samples(signed_wav)
        scaled = samples * 2.0  # will clip in _samples_to_wav
        wav_out = _samples_to_wav(scaled, sr)
        result = _verify_wav(wav_out)
        _print_result("Volume 2.0x (clipped)", result)

    def test_trim_start(self, signed_wav):
        samples, sr = _wav_to_samples(signed_wav)
        trim_samples = int(0.5 * sr)  # 0.5 seconds
        trimmed = samples[trim_samples:]
        wav_out = _samples_to_wav(trimmed, sr)
        result = _verify_wav(wav_out)
        _print_result("Trim first 0.5s", result)

    def test_trim_end(self, signed_wav):
        samples, sr = _wav_to_samples(signed_wav)
        trim_samples = int(0.5 * sr)
        trimmed = samples[:-trim_samples]
        wav_out = _samples_to_wav(trimmed, sr)
        result = _verify_wav(wav_out)
        _print_result("Trim last 0.5s", result)

    def test_prepend_silence(self, signed_wav):
        samples, sr = _wav_to_samples(signed_wav)
        silence = np.zeros(int(0.3 * sr))
        padded = np.concatenate([silence, samples])
        wav_out = _samples_to_wav(padded, sr)
        result = _verify_wav(wav_out)
        _print_result("Prepend 0.3s silence", result)

    def test_resample_telephone(self, signed_wav):
        """Simulate telephone: 16kHz -> 8kHz -> 16kHz."""
        samples, sr = _wav_to_samples(signed_wav)
        assert sr == SAMPLE_RATE

        # Downsample to 8kHz (take every other sample -- crude but fast)
        down = samples[::2]
        # Upsample back to 16kHz (linear interpolation)
        x_down = np.arange(len(down))
        x_up = np.linspace(0, len(down) - 1, len(samples))
        up = np.interp(x_up, x_down, down)

        wav_out = _samples_to_wav(up, sr)
        result = _verify_wav(wav_out)
        _print_result("Resample 16k->8k->16k", result)


# ---------------------------------------------------------------------------
# Summary test
# ---------------------------------------------------------------------------

class TestSurvivalSummary:
    """Run all transformations and print a consolidated results table."""

    def test_print_summary_table(self, signed_wav):
        """
        Not a pass/fail test -- always passes. Prints a summary table
        of watermark survival across all transformations.
        """
        results: list[tuple[str, dict]] = []

        # --- Baseline ---
        results.append(("Baseline (no transform)", _verify_wav(signed_wav)))

        # --- FFmpeg codecs ---
        codec_transforms: list[tuple[str, str, list[str]]] = [
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
                results.append((label, {"detected": None, "correlation": None, "confidence": None}))
                continue
            try:
                transformed = _ffmpeg_transcode(signed_wav, ext, args)
                results.append((label, _verify_wav(transformed)))
            except RuntimeError:
                results.append((label, {"detected": None, "correlation": None, "confidence": None}))

        # --- Signal transforms ---
        samples, sr = _wav_to_samples(signed_wav)
        rng = np.random.RandomState(42)

        def _add_noise(snr_db):
            sig_pow = np.mean(samples ** 2)
            noise_pow = sig_pow / (10 ** (snr_db / 10))
            n = rng.randn(len(samples)) * np.sqrt(noise_pow)
            return _samples_to_wav(samples + n, sr)

        results.append(("White noise SNR 30 dB", _verify_wav(_add_noise(30))))
        # Reset RNG for reproducibility
        rng = np.random.RandomState(42)
        results.append(("White noise SNR 20 dB", _verify_wav(_add_noise(20))))

        results.append(("Volume 0.5x", _verify_wav(_samples_to_wav(samples * 0.5, sr))))
        results.append(("Volume 2.0x (clipped)", _verify_wav(_samples_to_wav(samples * 2.0, sr))))

        trim_n = int(0.5 * sr)
        results.append(("Trim first 0.5s", _verify_wav(_samples_to_wav(samples[trim_n:], sr))))
        results.append(("Trim last 0.5s", _verify_wav(_samples_to_wav(samples[:-trim_n], sr))))

        silence = np.zeros(int(0.3 * sr))
        results.append(("Prepend 0.3s silence", _verify_wav(
            _samples_to_wav(np.concatenate([silence, samples]), sr)
        )))

        # Telephone resample
        down = samples[::2]
        x_down = np.arange(len(down))
        x_up = np.linspace(0, len(down) - 1, len(samples))
        up = np.interp(x_up, x_down, down)
        results.append(("Resample 16k->8k->16k", _verify_wav(_samples_to_wav(up, sr))))

        # --- Print table ---
        print("\n")
        print("=" * 78)
        print("  VOICESIGN CODEC SURVIVAL SUMMARY")
        print("=" * 78)
        hdr = f"  {'Transformation':<35s} | {'Detected':>8s} | {'Correlation':>11s} | {'Confidence':>10s}"
        print(hdr)
        print("  " + "-" * 35 + "-+-" + "-" * 8 + "-+-" + "-" * 11 + "-+-" + "-" * 10)

        survived = 0
        tested = 0

        for label, res in results:
            if res["detected"] is None:
                status = "SKIP"
                corr_str = "N/A"
                conf_str = "N/A"
            else:
                tested += 1
                det = res["detected"]
                if det:
                    survived += 1
                status = "YES" if det else "NO"
                corr_str = f"{res['correlation']:.6f}"
                conf_str = f"{res['confidence']:.2%}"
            print(f"  {label:<35s} | {status:>8s} | {corr_str:>11s} | {conf_str:>10s}")

        print("  " + "-" * 35 + "-+-" + "-" * 8 + "-+-" + "-" * 11 + "-+-" + "-" * 10)
        print(f"  Survived: {survived}/{tested} tested  |  Detection threshold: {DETECTION_THRESHOLD}")
        print("=" * 78)
        print()

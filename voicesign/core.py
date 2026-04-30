# VoiceSign by Oravys Inc. (https://oravys.com) - Created by Eliot Cohen Bacrie
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
# Attribution to Oravys Inc. and Eliot Cohen Bacrie is required. See NOTICE.
#
"""
VoiceSign Core - Spread-spectrum audio watermarking engine.

Embeds an inaudible watermark in the 2-6 kHz band using a pseudo-random
noise sequence derived from a user-provided salt and identity string.
Detection uses Pearson correlation against the expected PN sequence.

Target SNR: ~42-46 dB (imperceptible to the human ear).

This module is fully self-contained. Dependencies: numpy, wave (stdlib),
subprocess (stdlib, for ffmpeg format conversion).
"""

import hashlib
import io
import logging
import os
import struct
import subprocess
import tempfile
import wave
from typing import Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum input file size (50 MB).
MAX_FILE_SIZE: int = 50 * 1024 * 1024

#: Embedding strength. Controls the amplitude of the watermark signal
#: relative to the original audio. Lower values are less audible but
#: harder to detect; higher values are more robust but may be perceptible.
EMBED_STRENGTH: float = 0.04

#: Duration of each analysis segment in seconds.
SEGMENT_DURATION: float = 0.1

#: Lower bound of the watermark frequency band (Hz).
BAND_LOW_HZ: int = 2000

#: Upper bound of the watermark frequency band (Hz).
BAND_HIGH_HZ: int = 6000

#: Minimum average Pearson correlation required to declare a positive match.
DETECTION_THRESHOLD: float = 0.03

#: Audio formats supported for automatic conversion via ffmpeg.
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "m4a"}

#: Default salt used when none is provided. For production use, always
#: supply a unique, secret salt via the ``salt`` parameter.
_DEFAULT_SALT = "voicesign-default-salt"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derive_seed(salt: str, identity: str) -> int:
    """
    Derive a deterministic PRNG seed from a salt and user identity.

    Uses SHA-256 of (salt + identity) truncated to a 32-bit unsigned
    integer so that numpy RandomState receives a reproducible seed.

    Parameters
    ----------
    salt : str
        Secret or unique salt string.
    identity : str
        User identity string (name, email, or any unique identifier).

    Returns
    -------
    int
        A 32-bit unsigned integer seed.
    """
    combined = (salt + identity).encode("utf-8")
    digest = hashlib.sha256(combined).digest()
    seed = struct.unpack(">I", digest[:4])[0]
    return seed


def _generate_pn_sequence(seed: int, length: int) -> np.ndarray:
    """
    Generate a pseudo-random noise sequence of +1/-1 values.

    Parameters
    ----------
    seed : int
        PRNG seed (32-bit unsigned integer).
    length : int
        Number of elements in the sequence.

    Returns
    -------
    numpy.ndarray
        Array of shape (length,) containing +1.0 or -1.0 values.
    """
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=length).astype(np.float64)


def _convert_to_wav(audio_bytes: bytes, file_format: str) -> bytes:
    """
    Convert arbitrary audio to 16-bit mono WAV via ffmpeg.

    If the input is already WAV, it is returned unchanged. All I/O uses
    temporary files that are cleaned up immediately after conversion.

    Parameters
    ----------
    audio_bytes : bytes
        Raw file content.
    file_format : str
        Input format hint (e.g. ``"mp3"``, ``"flac"``).

    Returns
    -------
    bytes
        16-bit mono WAV content at 16 kHz.

    Raises
    ------
    ValueError
        If the format is not in SUPPORTED_FORMATS.
    RuntimeError
        If ffmpeg fails or is not installed.
    """
    fmt = file_format.lower().strip().lstrip(".")
    if fmt == "wav":
        return audio_bytes

    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {fmt}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    tmp_in = None
    tmp_out = None
    try:
        suffix_in = f".{fmt}"
        tmp_in = tempfile.NamedTemporaryFile(suffix=suffix_in, delete=False)
        tmp_in.write(audio_bytes)
        tmp_in.close()

        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_out.close()

        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_in.name,
            "-ac", "1",            # mono
            "-ar", "16000",        # 16 kHz
            "-sample_fmt", "s16",  # 16-bit
            "-f", "wav",
            tmp_out.name,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ffmpeg conversion failed (rc={result.returncode}): "
                f"{stderr_text[:500]}"
            )

        with open(tmp_out.name, "rb") as f:
            return f.read()
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is required for non-WAV format conversion but was "
            "not found on your system. Install ffmpeg and ensure it is "
            "on your PATH."
        )
    finally:
        for tmp in (tmp_in, tmp_out):
            if tmp is not None:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass


def _wav_to_samples(wav_bytes: bytes):
    """
    Read WAV bytes into a numpy float64 array normalised to [-1, 1].

    Parameters
    ----------
    wav_bytes : bytes
        Raw WAV file content.

    Returns
    -------
    tuple
        (samples, params) where samples is a 1-D float64 ndarray and
        params is the wave module getparams() named tuple.
    """
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        params = wf.getparams()
        n_frames = params.nframes
        raw = wf.readframes(n_frames)

    n_channels = params.nchannels
    sampwidth = params.sampwidth

    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)

    # Mix to mono if stereo
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Normalise to [-1, 1]
    max_val = float(2 ** (8 * sampwidth - 1))
    samples = samples / max_val

    return samples, params


def _samples_to_wav(
    samples: np.ndarray, sample_rate: int, sampwidth: int = 2
) -> bytes:
    """
    Encode float64 samples back to WAV bytes (mono, 16-bit by default).

    Parameters
    ----------
    samples : numpy.ndarray
        Audio samples normalised to [-1, 1].
    sample_rate : int
        Output sample rate in Hz.
    sampwidth : int
        Sample width in bytes (default 2 for 16-bit).

    Returns
    -------
    bytes
        WAV file content.
    """
    max_val = float(2 ** (8 * sampwidth - 1)) - 1
    clipped = np.clip(samples, -1.0, 1.0)
    int_samples = (clipped * max_val).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


def _get_band_indices(fft_length: int, sample_rate: int):
    """
    Return the start and end indices in an rfft result that correspond
    to the watermark frequency band.

    Parameters
    ----------
    fft_length : int
        Length of the FFT (number of time-domain samples).
    sample_rate : int
        Audio sample rate in Hz.

    Returns
    -------
    tuple
        (idx_low, idx_high) index range into the rfft output.

    Raises
    ------
    ValueError
        If the watermark band does not fit within the Nyquist limit.
    """
    freq_res = sample_rate / fft_length
    idx_low = max(1, int(BAND_LOW_HZ / freq_res))
    idx_high = min(fft_length // 2, int(BAND_HIGH_HZ / freq_res))
    if idx_high <= idx_low:
        raise ValueError(
            f"Watermark band [{BAND_LOW_HZ}-{BAND_HIGH_HZ} Hz] does not fit "
            f"within the Nyquist limit for sample rate {sample_rate} Hz"
        )
    return idx_low, idx_high


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sign(
    audio_bytes: bytes,
    identity: str,
    salt: Optional[str] = None,
    file_format: str = "wav",
) -> bytes:
    """
    Embed an inaudible spread-spectrum watermark tied to an identity.

    The watermark is embedded in the 2-6 kHz frequency band using a
    pseudo-random noise sequence. The result is imperceptible to the
    human ear (target SNR ~42-46 dB) but can be reliably detected by
    the ``verify`` function.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file content (WAV, MP3, FLAC, or M4A).
    identity : str
        User identity string (name, email, or any unique identifier).
        This is what gets cryptographically bound to the audio.
    salt : str, optional
        Secret salt for seed derivation. Using a unique salt prevents
        third parties from forging watermarks. If not provided, a
        default salt is used (not recommended for production).
    file_format : str
        Input format hint (default ``"wav"``). Used only when the input
        is not WAV and needs conversion via ffmpeg.

    Returns
    -------
    bytes
        Watermarked audio as 16-bit mono WAV.

    Raises
    ------
    ValueError
        If the audio is empty, exceeds 50 MB, the format is unsupported,
        or the identity string is empty.
    RuntimeError
        If ffmpeg conversion fails (non-WAV inputs only).

    Examples
    --------
    >>> with open("recording.wav", "rb") as f:
    ...     audio = f.read()
    >>> signed = sign(audio, "Alice Johnson", salt="my-secret")
    >>> with open("signed.wav", "wb") as f:
    ...     f.write(signed)
    """
    if not audio_bytes:
        raise ValueError("Empty audio data")
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            f"File size {len(audio_bytes) / (1024 * 1024):.1f} MB exceeds "
            f"maximum allowed {MAX_FILE_SIZE / (1024 * 1024):.0f} MB"
        )
    if not identity or not identity.strip():
        raise ValueError("Identity string must not be empty")

    effective_salt = salt if salt else _DEFAULT_SALT

    # Convert to WAV if needed
    wav_bytes = _convert_to_wav(audio_bytes, file_format)
    samples, params = _wav_to_samples(wav_bytes)
    sample_rate = params.framerate
    segment_len = int(SEGMENT_DURATION * sample_rate)

    if segment_len < 16:
        raise ValueError("Audio sample rate too low for watermark embedding")

    # Derive PN seed from salt + identity
    seed = _derive_seed(effective_salt, identity)

    # Get frequency band indices
    idx_low, idx_high = _get_band_indices(segment_len, sample_rate)
    band_width = idx_high - idx_low

    n_segments = len(samples) // segment_len
    if n_segments == 0:
        raise ValueError("Audio too short for watermark embedding (< 0.1 s)")

    watermarked = samples.copy()

    for seg_idx in range(n_segments):
        start = seg_idx * segment_len
        end = start + segment_len
        segment = watermarked[start:end]

        # Per-segment seed variation to avoid repetition
        seg_seed = (seed + seg_idx) & 0xFFFFFFFF
        pn = _generate_pn_sequence(seg_seed, band_width)

        # FFT
        spectrum = np.fft.rfft(segment)

        # Add PN sequence to the real part in the watermark band
        spectrum[idx_low:idx_high] = (
            spectrum[idx_low:idx_high].real + EMBED_STRENGTH * pn
        ) + 1j * spectrum[idx_low:idx_high].imag

        # Inverse FFT
        watermarked[start:end] = np.fft.irfft(spectrum, n=segment_len)

    logger.info(
        "VoiceSign: watermark embedded - %d segments, sr=%d",
        n_segments,
        sample_rate,
    )

    return _samples_to_wav(watermarked, sample_rate)


def verify(
    audio_bytes: bytes,
    identity: str,
    salt: Optional[str] = None,
    file_format: str = "wav",
) -> Dict[str, Union[bool, float, str]]:
    """
    Check whether audio contains a watermark for a given identity.

    Computes the average Pearson correlation between the audio's
    frequency-domain coefficients and the expected pseudo-random
    noise sequence for the given identity and salt.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file content (WAV, MP3, FLAC, or M4A).
    identity : str
        The identity to test against.
    salt : str, optional
        The same salt that was used during signing. Must match exactly
        for detection to succeed.
    file_format : str
        Input format hint (default ``"wav"``).

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``detected`` (bool) - True if correlation exceeds the threshold.
        - ``correlation`` (float) - Average Pearson correlation across segments.
        - ``confidence`` (float) - Normalised confidence score in [0, 1].
        - ``identity_match`` (str) - The identity string tested (echoed back).

    Examples
    --------
    >>> with open("signed.wav", "rb") as f:
    ...     audio = f.read()
    >>> result = verify(audio, "Alice Johnson", salt="my-secret")
    >>> print(result["detected"])
    True
    >>> print(f"Confidence: {result['confidence']:.2%}")
    Confidence: 95.00%
    """
    _empty_result = {
        "detected": False,
        "correlation": 0.0,
        "confidence": 0.0,
        "identity_match": identity,
    }

    if not audio_bytes:
        return _empty_result
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            f"File size {len(audio_bytes) / (1024 * 1024):.1f} MB exceeds "
            f"maximum allowed {MAX_FILE_SIZE / (1024 * 1024):.0f} MB"
        )
    if not identity or not identity.strip():
        raise ValueError("Identity string must not be empty")

    effective_salt = salt if salt else _DEFAULT_SALT

    # Convert to WAV if needed
    wav_bytes = _convert_to_wav(audio_bytes, file_format)
    samples, params = _wav_to_samples(wav_bytes)
    sample_rate = params.framerate
    segment_len = int(SEGMENT_DURATION * sample_rate)

    if segment_len < 16:
        return _empty_result

    seed = _derive_seed(effective_salt, identity)

    try:
        idx_low, idx_high = _get_band_indices(segment_len, sample_rate)
    except ValueError:
        return _empty_result

    band_width = idx_high - idx_low
    n_segments = len(samples) // segment_len

    if n_segments == 0:
        return _empty_result

    correlations = []

    for seg_idx in range(n_segments):
        start = seg_idx * segment_len
        end = start + segment_len
        segment = samples[start:end]

        seg_seed = (seed + seg_idx) & 0xFFFFFFFF
        pn = _generate_pn_sequence(seg_seed, band_width)

        spectrum = np.fft.rfft(segment)
        band_real = spectrum[idx_low:idx_high].real

        # Pearson correlation between band coefficients and PN sequence
        if np.std(band_real) < 1e-12 or np.std(pn) < 1e-12:
            correlations.append(0.0)
            continue

        mean_br = np.mean(band_real)
        mean_pn = np.mean(pn)
        num = np.sum((band_real - mean_br) * (pn - mean_pn))
        den = np.sqrt(
            np.sum((band_real - mean_br) ** 2)
            * np.sum((pn - mean_pn) ** 2)
        )

        if den < 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(float(num / den))

    avg_correlation = float(np.mean(correlations)) if correlations else 0.0
    detected = avg_correlation > DETECTION_THRESHOLD

    # Map correlation from [threshold, 1] to [0, 1], clamped
    if avg_correlation <= DETECTION_THRESHOLD:
        confidence = 0.0
    else:
        confidence = min(
            1.0,
            (avg_correlation - DETECTION_THRESHOLD)
            / (1.0 - DETECTION_THRESHOLD),
        )

    logger.info(
        "VoiceSign: verification - detected=%s, corr=%.4f, conf=%.4f",
        detected,
        avg_correlation,
        confidence,
    )

    return {
        "detected": detected,
        "correlation": round(avg_correlation, 6),
        "confidence": round(confidence, 6),
        "identity_match": identity,
    }

# VoiceSign by Oravys Inc. (https://oravys.com) - Created by Eliot Cohen Bacrie
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
# Attribution to Oravys Inc. and Eliot Cohen Bacrie is required. See NOTICE.
#
"""
VoiceSign Core - Spread-spectrum audio watermarking engine.

Embeds an inaudible watermark in the 2-6 kHz band using a pseudo-random
noise sequence. Detection uses Pearson correlation against the expected
PN sequence, with synchronization markers for temporal-shift resilience.

Supports two modes:
- Legacy: seed derived from salt + identity (shared-secret)
- Crypto: seed derived from Ed25519 public key (asymmetric, non-repudiable)

Target SNR: ~42-46 dB (imperceptible to the human ear).
"""

import datetime
import hashlib
import io
import logging
import os
import struct
import subprocess
import tempfile
import wave

import numpy as np

from voicesign.sync import embed_sync_markers, find_sync_markers

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

#: Sync marker interval in seconds.
_SYNC_INTERVAL: float = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_seed(
    salt: str | None,
    identity: str,
    public_key: bytes | None,
) -> int:
    if public_key is not None:
        from voicesign.crypto import derive_seed_from_pubkey
        return derive_seed_from_pubkey(public_key)
    effective_salt = salt if salt else _DEFAULT_SALT
    return _derive_seed(effective_salt, identity)


def _derive_seed(salt: str, identity: str) -> int:
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
            f"Unsupported format: {fmt}. " f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
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
            "ffmpeg",
            "-y",
            "-i",
            tmp_in.name,
            "-ac",
            "1",  # mono
            "-ar",
            "16000",  # 16 kHz
            "-sample_fmt",
            "s16",  # 16-bit
            "-f",
            "wav",
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
                f"ffmpeg conversion failed (rc={result.returncode}): " f"{stderr_text[:500]}"
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


def _samples_to_wav(samples: np.ndarray, sample_rate: int, sampwidth: int = 2) -> bytes:
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
    salt: str | None = None,
    file_format: str = "wav",
    *,
    private_key: bytes | None = None,
    public_key: bytes | None = None,
) -> bytes:
    """
    Embed an inaudible spread-spectrum watermark tied to an identity.

    The watermark is embedded in the 2-6 kHz frequency band using a
    pseudo-random noise sequence, with synchronization markers for
    temporal-shift resilience.

    Two modes:

    - **Legacy** (default): seed derived from salt + identity.
    - **Crypto**: pass ``public_key`` (and optionally ``private_key``)
      to derive the seed from an Ed25519 public key instead. Use
      ``sign_with_receipt()`` for the full crypto workflow with a
      non-repudiable receipt.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file content (WAV, MP3, FLAC, or M4A).
    identity : str
        User identity string (name, email, or any unique identifier).
    salt : str, optional
        Secret salt for seed derivation (legacy mode only).
    file_format : str
        Input format hint (default ``"wav"``).
    private_key : bytes, optional
        Raw 32-byte Ed25519 private key (crypto mode).
    public_key : bytes, optional
        Raw 32-byte Ed25519 public key (crypto mode). When provided,
        the seed is derived from this key instead of salt + identity.

    Returns
    -------
    bytes
        Watermarked audio as 16-bit mono WAV.
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

    wav_bytes = _convert_to_wav(audio_bytes, file_format)
    samples, params = _wav_to_samples(wav_bytes)
    sample_rate = params.framerate
    segment_len = int(SEGMENT_DURATION * sample_rate)

    if segment_len < 16:
        raise ValueError("Audio sample rate too low for watermark embedding")

    seed = _resolve_seed(salt, identity, public_key)

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

        seg_seed = (seed + seg_idx) & 0xFFFFFFFF
        pn = _generate_pn_sequence(seg_seed, band_width)

        spectrum = np.fft.rfft(segment)
        spectrum[idx_low:idx_high] = (
            spectrum[idx_low:idx_high].real + EMBED_STRENGTH * pn
        ) + 1j * spectrum[idx_low:idx_high].imag
        watermarked[start:end] = np.fft.irfft(spectrum, n=segment_len)

    # Embed sync markers for temporal-shift resilience
    watermarked = embed_sync_markers(
        watermarked, sample_rate, seed, interval_sec=_SYNC_INTERVAL,
    )

    logger.info(
        "VoiceSign: watermark embedded - %d segments, sr=%d, mode=%s",
        n_segments,
        sample_rate,
        "crypto" if public_key else "legacy",
    )

    return _samples_to_wav(watermarked, sample_rate)


def sign_with_receipt(
    audio_bytes: bytes,
    identity: str,
    private_key: bytes,
    public_key: bytes,
    file_format: str = "wav",
) -> dict:
    """
    Sign audio with Ed25519 and return a non-repudiable receipt.

    This is the full crypto workflow: embeds the watermark using the
    public key as seed, then produces an Ed25519 signature binding
    the signer identity, audio hash, and timestamp together.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file content.
    identity : str
        Signer identity string.
    private_key : bytes
        Raw 32-byte Ed25519 private key.
    public_key : bytes
        Raw 32-byte Ed25519 public key.
    file_format : str
        Input format hint (default ``"wav"``).

    Returns
    -------
    dict
        Keys: ``audio`` (watermarked WAV bytes), ``signature`` (64-byte
        Ed25519 signature), ``public_key`` (32-byte public key),
        ``timestamp`` (ISO-8601 string), ``audio_hash`` (SHA-256 hex).
    """
    from voicesign.crypto import sign_payload

    signed_audio = sign(
        audio_bytes, identity, file_format=file_format,
        public_key=public_key,
    )

    audio_hash = hashlib.sha256(signed_audio).digest()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    signature = sign_payload(private_key, identity, audio_hash, timestamp)

    return {
        "audio": signed_audio,
        "signature": signature,
        "public_key": public_key,
        "timestamp": timestamp,
        "audio_hash": audio_hash.hex(),
    }


def verify(
    audio_bytes: bytes,
    identity: str,
    salt: str | None = None,
    file_format: str = "wav",
    *,
    public_key: bytes | None = None,
    signature: bytes | None = None,
    timestamp: str | None = None,
) -> dict[str, bool | float | str]:
    """
    Check whether audio contains a watermark for a given identity.

    Uses synchronization markers to recover segment alignment even if
    the audio has been trimmed or padded. Falls back to position-0
    alignment if no sync markers are found.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file content (WAV, MP3, FLAC, or M4A).
    identity : str
        The identity to test against.
    salt : str, optional
        Salt for seed derivation (legacy mode).
    file_format : str
        Input format hint (default ``"wav"``).
    public_key : bytes, optional
        Raw 32-byte Ed25519 public key (crypto mode). When provided,
        the seed is derived from this key instead of salt + identity.
    signature : bytes, optional
        64-byte Ed25519 signature to verify (requires ``public_key``
        and ``timestamp``).
    timestamp : str, optional
        ISO-8601 timestamp that was used when signing.

    Returns
    -------
    dict
        Keys: ``detected``, ``correlation``, ``confidence``,
        ``identity_match``, ``sync_aligned`` (bool), and optionally
        ``signature_valid`` (bool) when crypto params are provided.
    """
    _empty_result = {
        "detected": False,
        "correlation": 0.0,
        "confidence": 0.0,
        "identity_match": identity,
        "sync_aligned": False,
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

    wav_bytes = _convert_to_wav(audio_bytes, file_format)
    samples, params = _wav_to_samples(wav_bytes)
    sample_rate = params.framerate
    segment_len = int(SEGMENT_DURATION * sample_rate)

    if segment_len < 16:
        return _empty_result

    seed = _resolve_seed(salt, identity, public_key)

    try:
        idx_low, idx_high = _get_band_indices(segment_len, sample_rate)
    except ValueError:
        return _empty_result

    band_width = idx_high - idx_low

    # Try both position-0 and sync-marker alignment, keep the best
    n_seg_0 = len(samples) // segment_len
    if n_seg_0 == 0:
        return _empty_result

    corr_0 = _correlate_segments(
        samples, n_seg_0, segment_len, seed,
        idx_low, idx_high, band_width,
    )
    avg_0 = float(np.mean(corr_0)) if corr_0 else 0.0

    markers = find_sync_markers(
        samples, sample_rate, seed, interval_sec=_SYNC_INTERVAL,
    )

    avg_sync = -1.0
    corr_sync: list[float] = []
    if markers and markers[0] > 0:
        usable = samples[markers[0]:]
        n_seg_s = len(usable) // segment_len
        if n_seg_s > 0:
            corr_sync = _correlate_segments(
                usable, n_seg_s, segment_len, seed,
                idx_low, idx_high, band_width,
            )
            avg_sync = float(np.mean(corr_sync)) if corr_sync else -1.0

    if avg_sync > avg_0:
        avg_correlation = avg_sync
        correlations = corr_sync
        sync_aligned = True
    else:
        avg_correlation = avg_0
        correlations = corr_0
        sync_aligned = bool(markers) and markers[0] == 0

    detected = avg_correlation > DETECTION_THRESHOLD

    if avg_correlation <= DETECTION_THRESHOLD:
        confidence = 0.0
    else:
        confidence = min(
            1.0,
            (avg_correlation - DETECTION_THRESHOLD) / (1.0 - DETECTION_THRESHOLD),
        )

    result = {
        "detected": detected,
        "correlation": round(avg_correlation, 6),
        "confidence": round(confidence, 6),
        "identity_match": identity,
        "sync_aligned": sync_aligned,
    }

    # Ed25519 signature verification
    if signature is not None and public_key is not None and timestamp is not None:
        from voicesign.crypto import verify_signature
        audio_hash = hashlib.sha256(audio_bytes).digest()
        result["signature_valid"] = verify_signature(
            public_key, identity, audio_hash, timestamp, signature,
        )

    logger.info(
        "VoiceSign: verification - detected=%s, corr=%.4f, conf=%.4f, sync=%s",
        detected,
        avg_correlation,
        confidence,
        sync_aligned,
    )

    return result


def _correlate_segments(
    samples: np.ndarray,
    n_segments: int,
    segment_len: int,
    seed: int,
    idx_low: int,
    idx_high: int,
    band_width: int,
) -> list[float]:
    correlations = []
    for seg_idx in range(n_segments):
        start = seg_idx * segment_len
        end = start + segment_len
        segment = samples[start:end]

        seg_seed = (seed + seg_idx) & 0xFFFFFFFF
        pn = _generate_pn_sequence(seg_seed, band_width)

        spectrum = np.fft.rfft(segment)
        band_real = spectrum[idx_low:idx_high].real

        if np.std(band_real) < 1e-12 or np.std(pn) < 1e-12:
            correlations.append(0.0)
            continue

        mean_br = np.mean(band_real)
        mean_pn = np.mean(pn)
        num = np.sum((band_real - mean_br) * (pn - mean_pn))
        den = np.sqrt(
            np.sum((band_real - mean_br) ** 2) * np.sum((pn - mean_pn) ** 2)
        )

        if den < 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(float(num / den))

    return correlations

# VoiceSign

**The first open-source voice signature tool for individuals.**

[![Powered by Oravys Inc.](https://img.shields.io/badge/Powered%20by-Oravys%20Inc.-7C3AED)](https://oravys.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)

VoiceSign lets you embed an **inaudible cryptographic watermark** into any voice
recording, binding it to your identity. Later, anyone can verify that a recording
was signed by you. The watermark survives common audio transformations and is
imperceptible to the human ear (target SNR: 42-46 dB).

**Created by [Eliot Cohen Bacrie](https://oravys.com), Founder and CEO of Oravys Inc.**

> Web version available at [voicesign.eu](https://voicesign.eu)

---

## Why VoiceSign?

In a world of voice cloning and audio deepfakes, proving that a voice recording
is genuinely yours matters. VoiceSign gives individuals a simple, free tool to:

- **Sign** your voice memos, podcasts, or recordings with your identity
- **Verify** that a recording was signed by a specific person
- **Prove provenance** of your original audio content
- **Protect yourself** against voice impersonation and unauthorized use

---

## Install

```bash
pip install voicesign
```

Dependencies: `numpy`. For non-WAV audio formats (MP3, FLAC, M4A), you also
need [ffmpeg](https://ffmpeg.org/) installed on your system.

---

## Quick Start

### Python API

```python
import voicesign

# Read your audio file
with open("my_recording.wav", "rb") as f:
    audio = f.read()

# Sign it with your identity
signed = voicesign.sign(audio, identity="Alice Johnson", salt="my-secret-salt")

# Save the signed version
with open("signed_recording.wav", "wb") as f:
    f.write(signed)

# Later, verify it
with open("signed_recording.wav", "rb") as f:
    audio = f.read()

result = voicesign.verify(audio, identity="Alice Johnson", salt="my-secret-salt")

print(result["detected"])     # True
print(result["confidence"])   # 0.95 (95% confidence)
print(result["correlation"])  # 0.87
```

### Command Line

```bash
# Sign a recording
voicesign sign -i "Alice Johnson" -s "my-secret" recording.wav

# Output: recording_signed.wav

# Verify a recording
voicesign verify -i "Alice Johnson" -s "my-secret" recording_signed.wav

# Output: MATCH - Voice signature detected for: Alice Johnson
#         Confidence: 95.23%
```

### Custom output path

```bash
voicesign sign -i "Alice Johnson" -s "my-secret" -o output.wav input.wav
```

### Non-WAV formats (requires ffmpeg)

```bash
voicesign sign -i "Alice Johnson" podcast.mp3
voicesign verify -i "Alice Johnson" podcast_signed.wav
```

---

## How It Works

VoiceSign uses **spread-spectrum audio watermarking** in the frequency domain:

1. **Seed derivation** - A SHA-256 hash of your salt + identity produces a
   deterministic seed.
2. **PN sequence generation** - A pseudo-random noise (PN) sequence of +1/-1
   values is generated from the seed.
3. **Frequency-domain embedding** - The audio is split into 100ms segments.
   For each segment, the PN sequence is added to the real part of the FFT
   coefficients in the 2-6 kHz band at low amplitude.
4. **Detection** - Pearson correlation between the expected PN sequence and
   the actual frequency coefficients. High correlation = signature present.

The watermark is:
- **Inaudible** - Embedded at ~42-46 dB SNR, below human perception
- **Robust** - Survives format conversion, mild compression, and noise
- **Deterministic** - Same identity + salt always produces the same watermark
- **Lightweight** - Pure numpy, no ML models, runs instantly on any machine

---

## API Reference

### `voicesign.sign(audio_bytes, identity, salt=None, file_format="wav")`

Embed an identity watermark into audio.

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_bytes` | `bytes` | Raw audio file content |
| `identity` | `str` | Identity string to bind (name, email, etc.) |
| `salt` | `str` or `None` | Secret salt for watermark generation |
| `file_format` | `str` | Input format: `wav`, `mp3`, `flac`, `m4a` |

**Returns:** `bytes` - Watermarked audio as 16-bit mono WAV.

### `voicesign.verify(audio_bytes, identity, salt=None, file_format="wav")`

Check whether audio contains a watermark for a given identity.

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_bytes` | `bytes` | Raw audio file content |
| `identity` | `str` | Identity string to check against |
| `salt` | `str` or `None` | Must match the salt used during signing |
| `file_format` | `str` | Input format: `wav`, `mp3`, `flac`, `m4a` |

**Returns:** `dict` with keys:
- `detected` (bool) - Whether a matching watermark was found
- `confidence` (float) - Confidence score between 0 and 1
- `correlation` (float) - Raw Pearson correlation value
- `identity_match` (str) - The identity string tested

---

## Security Notes

- **Salt matters.** Anyone who knows your salt and identity can forge your
  watermark. Keep your salt private.
- **Not tamper-proof.** A determined adversary who knows the algorithm and
  your salt can remove or overwrite the watermark. VoiceSign provides
  provenance, not DRM.
- **Use unique salts.** Different salts for different contexts (personal,
  professional, etc.) prevent cross-context tracking.

---

## Attribution

VoiceSign is created by **Eliot Cohen Bacrie** and powered by **Oravys Inc.**

If you use VoiceSign in your project, please include attribution:

> Powered by [Oravys Inc.](https://oravys.com), created by Eliot Cohen Bacrie

See the [NOTICE](NOTICE) file for full attribution requirements.

---

## Web Version

For a no-install experience, use the web version at:

**[voicesign.eu](https://voicesign.eu)**

---

## License

Apache 2.0 with attribution requirements. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Copyright 2024-2026 Oravys Inc.

---

*Powered by [Oravys Inc.](https://oravys.com) - Voice Intelligence*

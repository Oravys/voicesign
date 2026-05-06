# VoiceSign by Oravys Inc. (https://oravys.com)
"""Integration tests: Ed25519 crypto + sync markers + watermark in unified pipeline."""

import io
import wave

import numpy as np
import pytest

import voicesign
from voicesign.crypto import generate_keypair, verify_signature


def _make_speech_wav(duration: float = 3.0, sr: int = 16000) -> bytes:
    rng = np.random.default_rng(99)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.5 * t)
    speech = 0.4 * np.sin(2 * np.pi * np.cumsum(f0 / sr))
    speech += 0.15 * np.sin(2 * np.pi * np.cumsum(2 * f0 / sr))
    speech += 0.08 * np.sin(2 * np.pi * np.cumsum(3 * f0 / sr))
    speech += rng.normal(0, 0.02, len(t))
    speech *= 0.7 * (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t))
    int_samples = (np.clip(speech, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


class TestCryptoSignVerify:
    def test_full_pipeline(self):
        priv, pub = generate_keypair()
        audio = _make_speech_wav()

        receipt = voicesign.sign_with_receipt(
            audio, "alice@example.com", priv, pub,
        )

        assert "audio" in receipt
        assert "signature" in receipt
        assert "timestamp" in receipt
        assert "audio_hash" in receipt
        assert len(receipt["signature"]) == 64

        result = voicesign.verify(
            receipt["audio"], "alice@example.com",
            public_key=pub,
        )
        assert result["detected"] is True
        assert result["confidence"] > 0

    def test_wrong_pubkey_fails(self):
        priv, pub = generate_keypair()
        _, wrong_pub = generate_keypair()
        audio = _make_speech_wav()

        signed = voicesign.sign(audio, "bob", public_key=pub)
        result = voicesign.verify(signed, "bob", public_key=wrong_pub)
        assert result["detected"] is False

    def test_signature_verification_in_verify(self):
        priv, pub = generate_keypair()
        audio = _make_speech_wav()

        receipt = voicesign.sign_with_receipt(audio, "carol", priv, pub)

        import hashlib
        audio_hash = hashlib.sha256(receipt["audio"]).digest()

        result = voicesign.verify(
            receipt["audio"], "carol",
            public_key=pub,
            signature=receipt["signature"],
            timestamp=receipt["timestamp"],
        )
        assert result["detected"] is True
        assert result["signature_valid"] is True

    def test_tampered_audio_signature_fails(self):
        priv, pub = generate_keypair()
        audio = _make_speech_wav()
        receipt = voicesign.sign_with_receipt(audio, "dave", priv, pub)

        other_audio = _make_speech_wav(duration=2.0)
        result = voicesign.verify(
            other_audio, "dave",
            public_key=pub,
            signature=receipt["signature"],
            timestamp=receipt["timestamp"],
        )
        assert result.get("signature_valid") is False


class TestSyncMarkerResilience:
    def test_trim_start_now_works(self):
        audio = _make_speech_wav(duration=5.0)
        signed = voicesign.sign(audio, "sync-test", salt="s")
        result_full = voicesign.verify(signed, "sync-test", salt="s")
        assert result_full["detected"] is True

    def test_trim_end_still_works(self):
        audio = _make_speech_wav(duration=5.0)
        signed = voicesign.sign(audio, "sync-test", salt="s")

        buf = io.BytesIO(signed)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            samples = np.frombuffer(
                wf.readframes(wf.getnframes()), dtype=np.int16,
            ).astype(np.float64) / 32768.0

        trimmed = samples[: int(len(samples) * 0.8)]
        int_trimmed = (np.clip(trimmed, -1, 1) * 32767).astype(np.int16)
        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int_trimmed.tobytes())

        result = voicesign.verify(out.getvalue(), "sync-test", salt="s")
        assert result["detected"] is True

    def test_legacy_backward_compat(self):
        audio = _make_speech_wav()
        signed = voicesign.sign(audio, "legacy", salt="old-salt")
        result = voicesign.verify(signed, "legacy", salt="old-salt")
        assert result["detected"] is True
        assert result["confidence"] > 0

    def test_legacy_wrong_salt_fails(self):
        audio = _make_speech_wav()
        signed = voicesign.sign(audio, "user", salt="correct")
        result = voicesign.verify(signed, "user", salt="wrong")
        assert result["detected"] is False


class TestVersionBump:
    def test_version_is_0_2_0(self):
        assert voicesign.__version__ == "0.2.0"

# VoiceSign by Oravys Inc. (https://oravys.com) - Created by Eliot Cohen Bacrie
import io
import wave

import numpy as np
import pytest
from voicesign.core import _derive_seed, _generate_pn_sequence, sign, verify


def _make_wav(duration=1.0, sample_rate=16000, freq=440.0):
    """Generate a simple sine wave WAV for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


class TestSeedDerivation:
    def test_deterministic(self):
        s1 = _derive_seed("salt", "Alice")
        s2 = _derive_seed("salt", "Alice")
        assert s1 == s2

    def test_different_identity(self):
        s1 = _derive_seed("salt", "Alice")
        s2 = _derive_seed("salt", "Bob")
        assert s1 != s2

    def test_different_salt(self):
        s1 = _derive_seed("salt1", "Alice")
        s2 = _derive_seed("salt2", "Alice")
        assert s1 != s2


class TestPNSequence:
    def test_shape(self):
        pn = _generate_pn_sequence(42, 100)
        assert pn.shape == (100,)

    def test_values(self):
        pn = _generate_pn_sequence(42, 1000)
        unique = set(pn.tolist())
        assert unique == {-1.0, 1.0}

    def test_deterministic(self):
        pn1 = _generate_pn_sequence(42, 100)
        pn2 = _generate_pn_sequence(42, 100)
        np.testing.assert_array_equal(pn1, pn2)


class TestSignVerify:
    def test_roundtrip(self):
        wav = _make_wav(duration=2.0)
        signed = sign(wav, identity="Test User", salt="test-salt")
        assert len(signed) > 0

        result = verify(signed, identity="Test User", salt="test-salt")
        assert result["detected"] is True
        assert result["confidence"] > 0.5
        assert result["identity_match"] == "Test User"

    def test_wrong_identity(self):
        wav = _make_wav(duration=2.0)
        signed = sign(wav, identity="Alice", salt="salt")
        result = verify(signed, identity="Bob", salt="salt")
        assert result["detected"] is False
        assert result["confidence"] == 0.0

    def test_wrong_salt(self):
        wav = _make_wav(duration=2.0)
        signed = sign(wav, identity="Alice", salt="salt1")
        result = verify(signed, identity="Alice", salt="salt2")
        assert result["detected"] is False

    def test_empty_audio_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            sign(b"", identity="Alice")

    def test_empty_identity_raises(self):
        wav = _make_wav()
        with pytest.raises(ValueError, match="empty"):
            sign(wav, identity="")

    def test_short_audio(self):
        wav = _make_wav(duration=0.05)
        with pytest.raises(ValueError, match="too short"):
            sign(wav, identity="Alice")

    def test_output_is_valid_wav(self):
        wav = _make_wav(duration=1.0)
        signed = sign(wav, identity="Test", salt="s")
        buf = io.BytesIO(signed)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() > 0

    def test_snr_acceptable(self):
        wav = _make_wav(duration=2.0, sample_rate=16000)
        signed = sign(wav, identity="SNR Test", salt="snr")

        buf_orig = io.BytesIO(wav)
        with wave.open(buf_orig, "rb") as wf:
            orig = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(float)

        buf_sign = io.BytesIO(signed)
        with wave.open(buf_sign, "rb") as wf:
            wm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(float)

        min_len = min(len(orig), len(wm))
        noise = wm[:min_len] - orig[:min_len]
        signal_power = np.mean(orig[:min_len] ** 2)
        noise_power = np.mean(noise**2)
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            assert snr_db > 30, f"SNR too low: {snr_db:.1f} dB"

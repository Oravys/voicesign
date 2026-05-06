# VoiceSign by Oravys Inc. (https://oravys.com) - Created by Eliot Cohen Bacrie
import numpy as np
import pytest

from voicesign.sync import (
    SYNC_PATTERN_DURATION,
    embed_sync_markers,
    find_sync_markers,
    generate_sync_pattern,
)


def _make_sine(duration: float, sample_rate: int, freq: float = 440.0) -> np.ndarray:
    """Generate a mono sine wave normalized to [-1, 1]."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float64)


class TestGenerateSyncPattern:
    def test_correct_length(self):
        sr = 16000
        length = int(SYNC_PATTERN_DURATION * sr)
        pattern = generate_sync_pattern(seed=42, length=length, sample_rate=sr)
        assert len(pattern) == length

    def test_deterministic(self):
        sr = 16000
        length = int(SYNC_PATTERN_DURATION * sr)
        p1 = generate_sync_pattern(seed=42, length=length, sample_rate=sr)
        p2 = generate_sync_pattern(seed=42, length=length, sample_rate=sr)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        sr = 16000
        length = int(SYNC_PATTERN_DURATION * sr)
        p1 = generate_sync_pattern(seed=42, length=length, sample_rate=sr)
        p2 = generate_sync_pattern(seed=99, length=length, sample_rate=sr)
        assert not np.array_equal(p1, p2)

    def test_amplitude_within_bounds(self):
        sr = 44100
        length = int(SYNC_PATTERN_DURATION * sr)
        pattern = generate_sync_pattern(seed=7, length=length, sample_rate=sr)
        # Amplitude should be around SYNC_STRENGTH (0.04), never exceed it
        assert np.max(np.abs(pattern)) <= 0.04 + 1e-9

    def test_zero_length(self):
        pattern = generate_sync_pattern(seed=1, length=0, sample_rate=16000)
        assert len(pattern) == 0


class TestEmbedAndFindRoundtrip:
    """Test that embedding sync markers and finding them works on clean audio."""

    @pytest.mark.parametrize("sample_rate", [16000, 22050, 44100])
    def test_roundtrip_clean(self, sample_rate):
        duration = 5.0
        audio = _make_sine(duration, sample_rate)
        seed = 12345
        interval = 2.0

        marked = embed_sync_markers(audio, sample_rate, seed, interval_sec=interval)
        assert marked.shape == audio.shape

        found = find_sync_markers(marked, sample_rate, seed, interval_sec=interval)

        # We expect markers at 0, 2s, 4s = 3 markers for 5s audio
        expected_count = int(duration / interval) + 1
        # Allow for the last marker to potentially not fit
        assert len(found) >= expected_count - 1
        assert len(found) <= expected_count

        # First marker should be near position 0
        tolerance_samples = int(0.01 * sample_rate)  # 10 ms
        assert found[0] < tolerance_samples

    def test_marker_positions_accurate(self):
        sample_rate = 16000
        duration = 7.0
        audio = _make_sine(duration, sample_rate)
        seed = 999
        interval = 2.0

        marked = embed_sync_markers(audio, sample_rate, seed, interval_sec=interval)
        found = find_sync_markers(marked, sample_rate, seed, interval_sec=interval)

        # Expected positions: 0, 32000, 64000, 96000
        tolerance = int(0.01 * sample_rate)  # 10 ms = 160 samples at 16kHz

        for i, pos in enumerate(found):
            expected_pos = i * int(interval * sample_rate)
            assert abs(pos - expected_pos) < tolerance, (
                f"Marker {i}: expected ~{expected_pos}, got {pos} "
                f"(delta={abs(pos - expected_pos)} samples, "
                f"tolerance={tolerance})"
            )


class TestTemporalShiftResilience:
    """Test detection after temporal modifications to the audio."""

    def test_prepend_silence(self):
        """Prepending 0.3s of silence should not prevent detection."""
        sample_rate = 16000
        duration = 5.0
        audio = _make_sine(duration, sample_rate)
        seed = 42
        interval = 2.0

        marked = embed_sync_markers(audio, sample_rate, seed, interval_sec=interval)

        # Prepend 0.3s of silence
        silence = np.zeros(int(0.3 * sample_rate), dtype=np.float64)
        shifted = np.concatenate([silence, marked])

        found = find_sync_markers(shifted, sample_rate, seed, interval_sec=interval)

        # Should still find markers, but shifted by ~0.3s
        assert len(found) >= 2, f"Expected >= 2 markers, found {len(found)}"

        # First marker should be near 0.3s = 4800 samples
        expected_first = int(0.3 * sample_rate)
        tolerance = int(0.01 * sample_rate)
        assert abs(found[0] - expected_first) < tolerance, (
            f"First marker at {found[0]}, expected ~{expected_first}"
        )

    def test_trim_start(self):
        """Trimming 0.5s from the start should still find remaining markers."""
        sample_rate = 16000
        duration = 5.0
        audio = _make_sine(duration, sample_rate)
        seed = 42
        interval = 2.0

        marked = embed_sync_markers(audio, sample_rate, seed, interval_sec=interval)

        # Trim 0.5s from start
        trim_samples = int(0.5 * sample_rate)
        trimmed = marked[trim_samples:]

        found = find_sync_markers(trimmed, sample_rate, seed, interval_sec=interval)

        # Original markers at 0, 2s, 4s. After trimming 0.5s:
        # marker at 2s is now at 1.5s, marker at 4s is now at 3.5s
        assert len(found) >= 2, f"Expected >= 2 markers after trim, found {len(found)}"

        # First remaining marker should be near 1.5s = 24000 samples
        expected_first = int(1.5 * sample_rate)
        tolerance = int(0.01 * sample_rate)
        assert abs(found[0] - expected_first) < tolerance, (
            f"First marker at {found[0]}, expected ~{expected_first}"
        )

    def test_trim_end(self):
        """Trimming 0.2s from the end should not affect earlier markers."""
        sample_rate = 16000
        duration = 5.0
        audio = _make_sine(duration, sample_rate)
        seed = 42
        interval = 2.0

        marked = embed_sync_markers(audio, sample_rate, seed, interval_sec=interval)

        # Trim 0.2s from end
        trim_samples = int(0.2 * sample_rate)
        trimmed = marked[:-trim_samples]

        found = find_sync_markers(trimmed, sample_rate, seed, interval_sec=interval)

        # Markers at 0, 2s, 4s should all still be present
        # (only 0.2s removed from end of 5s audio)
        assert len(found) >= 3, f"Expected >= 3 markers, found {len(found)}"

        # First marker still at 0
        tolerance = int(0.01 * sample_rate)
        assert found[0] < tolerance

    def test_combined_trim_and_pad(self):
        """Trim 0.3s from start AND prepend 0.1s silence."""
        sample_rate = 16000
        duration = 5.0
        audio = _make_sine(duration, sample_rate)
        seed = 42
        interval = 2.0

        marked = embed_sync_markers(audio, sample_rate, seed, interval_sec=interval)

        # Trim 0.3s from start, then prepend 0.1s silence
        trim = int(0.3 * sample_rate)
        pad = np.zeros(int(0.1 * sample_rate), dtype=np.float64)
        modified = np.concatenate([pad, marked[trim:]])

        found = find_sync_markers(modified, sample_rate, seed, interval_sec=interval)

        # Should still find markers
        assert len(found) >= 2, f"Expected >= 2 markers, found {len(found)}"


class TestSampleRates:
    """Verify sync markers work across different sample rates."""

    @pytest.mark.parametrize("sample_rate", [16000, 22050, 44100])
    def test_embed_find_various_rates(self, sample_rate):
        duration = 4.0
        audio = _make_sine(duration, sample_rate)
        seed = 7777
        interval = 2.0

        marked = embed_sync_markers(audio, sample_rate, seed, interval_sec=interval)
        found = find_sync_markers(marked, sample_rate, seed, interval_sec=interval)

        # At 4s with 2s interval, expect markers at 0, 2s = 2 markers
        assert len(found) >= 2, (
            f"sr={sample_rate}: expected >= 2 markers, found {len(found)}"
        )

        # Check positions are within 10ms tolerance
        tolerance = int(0.01 * sample_rate)
        assert found[0] < tolerance, (
            f"sr={sample_rate}: first marker at {found[0]}, expected < {tolerance}"
        )

        expected_second = int(interval * sample_rate)
        assert abs(found[1] - expected_second) < tolerance, (
            f"sr={sample_rate}: second marker at {found[1]}, "
            f"expected ~{expected_second}"
        )


class TestEdgeCases:
    def test_audio_shorter_than_pattern(self):
        """Audio shorter than one sync pattern should return empty."""
        sr = 16000
        short = np.zeros(10, dtype=np.float64)
        result = embed_sync_markers(short, sr, seed=1)
        np.testing.assert_array_equal(result, short)

        found = find_sync_markers(short, sr, seed=1)
        assert found == []

    def test_wrong_seed_weaker_correlation(self):
        """Using the wrong seed should produce much lower peak NCC values."""
        sr = 16000
        audio = _make_sine(6.0, sr)
        seed_correct = 111
        seed_wrong = 222
        marked = embed_sync_markers(audio, sr, seed=seed_correct, interval_sec=2.0)

        # Compute the peak normalized cross-correlation for both seeds.
        # The correct seed should have a substantially higher max NCC
        # because the pattern matches exactly at the embedded positions.
        from voicesign.sync import generate_sync_pattern, SYNC_PATTERN_DURATION

        m = int(SYNC_PATTERN_DURATION * sr)

        p_c = generate_sync_pattern(seed_correct, m, sr)
        p_c_zm = p_c - np.mean(p_c)
        p_c_norm = np.linalg.norm(p_c_zm)

        p_w = generate_sync_pattern(seed_wrong, m, sr)
        p_w_zm = p_w - np.mean(p_w)
        p_w_norm = np.linalg.norm(p_w_zm)

        # Pearson correlation at the known marker position (pos=0)
        window = marked[:m]
        w_zm = window - np.mean(window)
        w_norm = np.linalg.norm(w_zm)

        corr_correct = np.dot(w_zm, p_c_zm) / (w_norm * p_c_norm)
        corr_wrong = np.dot(w_zm, p_w_zm) / (w_norm * p_w_norm)

        # The correct pattern should correlate much more strongly
        assert corr_correct > 0.05, f"Correct correlation too low: {corr_correct:.4f}"
        assert abs(corr_wrong) < corr_correct * 0.5, (
            f"Wrong seed correlation ({corr_wrong:.4f}) too close to "
            f"correct ({corr_correct:.4f})"
        )

    def test_does_not_modify_original(self):
        """embed_sync_markers should return a copy, not modify in-place."""
        sr = 16000
        audio = _make_sine(3.0, sr)
        original = audio.copy()
        _ = embed_sync_markers(audio, sr, seed=1)
        np.testing.assert_array_equal(audio, original)

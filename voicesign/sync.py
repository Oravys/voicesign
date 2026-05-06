# VoiceSign by Oravys Inc. (https://oravys.com) - Created by Eliot Cohen Bacrie
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
# Attribution to Oravys Inc. and Eliot Cohen Bacrie is required. See NOTICE.
#
"""
VoiceSign Sync - Synchronization markers for temporal-shift resilience.

Embeds short, detectable sync patterns at regular intervals so that the
watermark segment grid can be recovered even after the audio has been
trimmed, padded, or temporally shifted.

The sync pattern operates in the 3-4 kHz sub-band to avoid interference
with the main watermark band (2-6 kHz uses the full band, but the sync
pattern is a narrow chirp that occupies only a thin slice and is added
*on top* of the existing signal rather than replacing FFT coefficients).

Detection uses normalized cross-correlation in the time domain.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Duration of the sync pattern in seconds.
SYNC_PATTERN_DURATION: float = 0.05  # 50 ms

#: Lower bound of the sync chirp frequency (Hz).
SYNC_FREQ_LOW: int = 3000

#: Upper bound of the sync chirp frequency (Hz).
SYNC_FREQ_HIGH: int = 4000

#: Amplitude of the sync pattern relative to full scale.
SYNC_STRENGTH: float = 0.04

#: Minimum normalized correlation to accept a sync marker detection.
SYNC_DETECTION_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_sync_pattern(seed: int, length: int, sample_rate: int) -> np.ndarray:
    """
    Generate a deterministic synchronization pattern.

    The pattern is a pseudo-random phase-modulated chirp in the 3-4 kHz
    sub-band.  It is short (~50 ms worth of samples at the given rate),
    deterministic for a given seed, and designed to produce a sharp
    auto-correlation peak for reliable detection via cross-correlation.

    Parameters
    ----------
    seed : int
        PRNG seed that determines the pattern.  Must match between
        embedding and detection.
    length : int
        Number of samples in the pattern.  Typically
        ``int(SYNC_PATTERN_DURATION * sample_rate)``.
    sample_rate : int
        Audio sample rate in Hz.

    Returns
    -------
    numpy.ndarray
        1-D float64 array of shape ``(length,)`` with values scaled to
        ``SYNC_STRENGTH``.
    """
    if length <= 0:
        return np.array([], dtype=np.float64)

    rng = np.random.RandomState(seed & 0xFFFFFFFF)

    # Time vector for the pattern
    t = np.arange(length, dtype=np.float64) / sample_rate

    # Linear chirp from SYNC_FREQ_LOW to SYNC_FREQ_HIGH
    # Instantaneous frequency: f(t) = f_low + (f_high - f_low) * t / T
    # Phase: phi(t) = 2*pi * integral of f(t) dt
    #       = 2*pi * (f_low * t + (f_high - f_low) * t^2 / (2*T))
    T = length / sample_rate  # total duration
    if T <= 0:
        return np.zeros(length, dtype=np.float64)

    phase = 2.0 * np.pi * (
        SYNC_FREQ_LOW * t
        + (SYNC_FREQ_HIGH - SYNC_FREQ_LOW) * t ** 2 / (2.0 * T)
    )

    chirp = np.cos(phase)

    # Modulate with a pseudo-random sign sequence to spread the spectrum
    # and make the pattern unique per seed.  Use chunks of ~10 samples
    # to keep the pattern smooth enough for correlation.
    chunk_size = max(1, length // 16)
    signs = rng.choice([-1.0, 1.0], size=(length // chunk_size) + 1)
    sign_expanded = np.repeat(signs, chunk_size)[:length]

    pattern = chirp * sign_expanded * SYNC_STRENGTH

    return pattern.astype(np.float64)


def embed_sync_markers(
    samples: np.ndarray,
    sample_rate: int,
    seed: int,
    interval_sec: float = 2.0,
) -> np.ndarray:
    """
    Embed synchronization markers into audio samples.

    A sync pattern is added at sample position 0 and then every
    ``interval_sec`` seconds.  The pattern is *added* to the existing
    audio (not replaced), preserving the original content.

    Parameters
    ----------
    samples : numpy.ndarray
        1-D float64 audio samples (mono, normalized to [-1, 1]).
    sample_rate : int
        Audio sample rate in Hz.
    seed : int
        PRNG seed for the sync pattern (must match at detection time).
    interval_sec : float
        Interval between consecutive sync markers in seconds.

    Returns
    -------
    numpy.ndarray
        Copy of ``samples`` with sync markers embedded.
    """
    pattern_len = int(SYNC_PATTERN_DURATION * sample_rate)
    if pattern_len <= 0 or len(samples) < pattern_len:
        return samples.copy()

    pattern = generate_sync_pattern(seed, pattern_len, sample_rate)
    interval_samples = int(interval_sec * sample_rate)
    if interval_samples <= 0:
        return samples.copy()

    result = samples.copy()
    pos = 0
    while pos + pattern_len <= len(result):
        result[pos : pos + pattern_len] += pattern
        pos += interval_samples

    return result


def find_sync_markers(
    samples: np.ndarray,
    sample_rate: int,
    seed: int,
    interval_sec: float = 2.0,
) -> list:
    """
    Detect synchronization markers in audio via cross-correlation.

    Searches the entire audio for occurrences of the expected sync
    pattern.  Works even if the audio has been trimmed, padded, or
    shifted -- the first detected marker may not be at position 0.

    The function computes the normalized cross-correlation between the
    audio and the expected pattern, then picks peaks that exceed
    ``SYNC_DETECTION_THRESHOLD`` times the maximum correlation value.

    Parameters
    ----------
    samples : numpy.ndarray
        1-D float64 audio samples (mono).
    sample_rate : int
        Audio sample rate in Hz.
    seed : int
        Same seed used during embedding.
    interval_sec : float
        Expected interval between markers (used for minimum peak
        spacing during detection -- peaks closer than half the interval
        are suppressed).

    Returns
    -------
    list of int
        Sample positions where sync markers were detected, sorted in
        ascending order.
    """
    pattern_len = int(SYNC_PATTERN_DURATION * sample_rate)
    if pattern_len <= 0 or len(samples) < pattern_len:
        return []

    pattern = generate_sync_pattern(seed, pattern_len, sample_rate)

    # Zero-mean normalized cross-correlation (Pearson correlation per
    # sliding window).  This is more discriminative than simple cosine
    # similarity because it removes the DC component from both the
    # pattern and each audio window before correlating.
    n = len(samples)
    m = pattern_len
    n_positions = n - m + 1
    if n_positions <= 0:
        return []

    # Zero-mean the pattern once
    pattern_zm = pattern - np.mean(pattern)
    pattern_zm_norm = np.sqrt(np.sum(pattern_zm ** 2))
    if pattern_zm_norm < 1e-12:
        return []

    # Compute sliding window means using cumulative sums
    cumsum = np.cumsum(samples)
    cumsum_sq = np.cumsum(samples ** 2)

    # Window sums for positions 0..n_positions-1
    window_sum = np.empty(n_positions, dtype=np.float64)
    window_sum[0] = cumsum[m - 1]
    window_sum[1:] = cumsum[m:] - cumsum[:n_positions - 1]

    window_sum_sq = np.empty(n_positions, dtype=np.float64)
    window_sum_sq[0] = cumsum_sq[m - 1]
    window_sum_sq[1:] = cumsum_sq[m:] - cumsum_sq[:n_positions - 1]

    window_means = window_sum / m

    # Cross-correlation of samples with the zero-meaned pattern
    raw_corr = np.correlate(samples, pattern_zm, mode="valid")

    # Subtract the contribution of the window mean:
    # sum((x_i - mean_x) * p_zm_i) = sum(x_i * p_zm_i) - mean_x * sum(p_zm_i)
    # Since pattern_zm is zero-mean, sum(p_zm_i) = 0, so raw_corr is
    # already the zero-mean cross-correlation numerator.

    # Window variance: var = sum(x^2)/m - mean^2
    # Window std norm: sqrt(sum((x_i - mean)^2)) = sqrt(sum(x^2) - m*mean^2)
    window_var_sum = window_sum_sq - m * window_means ** 2
    window_zm_norms = np.sqrt(np.maximum(window_var_sum, 1e-24))

    normalized_corr = raw_corr / (window_zm_norms * pattern_zm_norm)

    # Find peaks above threshold
    if len(normalized_corr) == 0:
        return []

    max_corr = np.max(normalized_corr)
    if max_corr < 1e-12:
        return []

    # Absolute threshold: correlation must be at least SYNC_DETECTION_THRESHOLD
    # of the global maximum
    abs_threshold = SYNC_DETECTION_THRESHOLD * max_corr

    # Minimum spacing between detected peaks (half the interval)
    min_spacing = int(interval_sec * sample_rate * 0.5)

    # Find all positions above threshold
    candidates = np.where(normalized_corr >= abs_threshold)[0]
    if len(candidates) == 0:
        return []

    # Greedy peak picking: take the highest peak, suppress neighbors,
    # repeat.
    corr_at_candidates = normalized_corr[candidates]
    order = np.argsort(-corr_at_candidates)  # descending by correlation

    selected = []
    suppressed = set()

    for idx in order:
        pos = int(candidates[idx])
        if pos in suppressed:
            continue
        selected.append(pos)
        # Suppress all candidates within min_spacing
        for other_idx in range(len(candidates)):
            if abs(int(candidates[other_idx]) - pos) < min_spacing:
                suppressed.add(int(candidates[other_idx]))

    selected.sort()
    return selected

"""FFT-based ephys metric computation."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.fft

from aind_ephys_ibl_gui_conversion.types import (
    BlockMetrics,
    ExperimentBlock,
    ShankChannels,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COHERENCE_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4),
    "theta": (4, 12),
    "alpha": (12, 30),
    "beta": (30, 100),
    "gamma": (100, 300),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _parseval_rms(
    X: np.ndarray,
    freq_mask: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Band-limited RMS via Parseval's theorem.

    Parameters
    ----------
    X : np.ndarray
        One-sided FFT coefficients, shape ``(n_freqs, n_channels)``.
    freq_mask : np.ndarray
        Boolean mask selecting frequency bins for the band.
    n_samples : int
        Number of time-domain samples in the original window.

    Returns
    -------
    np.ndarray
        RMS per channel, shape ``(n_channels,)``.
    """
    # For one-sided FFT of a real signal, interior bins (not DC or
    # Nyquist) are doubled to account for the missing negative freqs.
    # Our LFP and AP masks never include DC or Nyquist, so all
    # selected bins get the 2x factor.
    power = np.sum(np.abs(X[freq_mask]) ** 2, axis=0)
    return np.sqrt(2.0 * power / n_samples**2)


def _normalize_spectral_estimates(
    Sij: dict[str, np.ndarray],
    psd_accum: np.ndarray,
    shank_info: dict,
    n_windows: int,
    window_power: float,
    fs_lfp: float,
    lfp_mask: np.ndarray,
    lfp_freqs: np.ndarray,
    bands: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Normalize accumulated PSD and cross-spectral density.

    Parameters
    ----------
    Sij : dict[str, np.ndarray]
        Accumulated cross-spectral density per band.
    psd_accum : np.ndarray
        Accumulated PSD power, shape ``(n_lfp_bins, n_channels)``.
    shank_info : dict
        Per-shank channel indices and depth ordering.
    n_windows : int
        Number of FFT windows that were accumulated.
    window_power : float
        Sum of squared Hann window coefficients.
    fs_lfp : float
        LFP sampling frequency in Hz.
    lfp_mask : np.ndarray
        Boolean mask for LFP frequency bins.
    lfp_freqs : np.ndarray
        Full array of LFP frequencies from rfftfreq.
    bands : dict[str, tuple[float, float]]
        Frequency bands used for coherency.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict, dict]
        ``(psd_power, psd_freqs, correlation, coherency)``
    """
    # --- Normalize PSD ---
    psd_full = (psd_accum / (n_windows * window_power * fs_lfp)).astype(
        np.float32
    )
    # One-sided doubling (all bins are interior -- DC and Nyquist
    # are excluded by the LFP mask)
    psd_full *= 2.0
    psd_power = psd_full
    psd_freqs = lfp_freqs[lfp_mask].astype(np.float32)

    # --- Normalize coherency per shank ---
    correlation = {}
    complex_coherency = {}
    for band_name in bands:
        S = Sij[band_name] / n_windows
        for group_val, info in shank_info.items():
            idx = info["indices"]
            order = info["depth_order"]
            S_shank = S[np.ix_(idx[order], idx[order])]
            d = np.sqrt(np.maximum(np.real(np.diag(S_shank)), 0.0))
            d[d == 0] = 1.0
            coherency = S_shank / np.outer(d, d)
            shank_index = int(group_val) + 1
            correlation[(band_name, shank_index)] = np.real(coherency).astype(
                np.float32
            )
            complex_coherency[(band_name, shank_index)] = coherency.astype(
                np.complex64
            )

    return psd_power, psd_freqs, correlation, complex_coherency


def _compute_all_metrics(  # noqa: C901
    block: ExperimentBlock,
    window_interval: float = 30.0,
    window_duration: float = 4.0,
    ap_rms_duration: float = 1.0,
    ap_freq_min: float = 300.0,
    lfp_freq_range: tuple[float, float] = (1.0, 300.0),
    bands: dict[str, tuple[float, float]] | None = None,
    min_coherency_windows: int = 20,
) -> BlockMetrics:
    """Compute all ephys metrics from sparse windows.

    For each window: read raw data, compute AP RMS from a short
    FFT at full sample rate, decimate to LFP rate, then compute
    LFP RMS + PSD + cross-spectral density for coherency.

    Parameters
    ----------
    block : ExperimentBlock
        Experiment block containing recording and optional LFP.
    window_interval : float
        Maximum interval in seconds between windows.  Shortened
        automatically for short recordings to ensure at least
        ``min_coherency_windows`` segments.
    window_duration : float
        Duration of each window in seconds (used for LFP metrics).
    ap_rms_duration : float
        Duration in seconds of the sub-window used for AP RMS
        at the native sample rate.
    ap_freq_min : float
        Lower edge of the AP band in Hz.
    lfp_freq_range : tuple[float, float]
        ``(low, high)`` frequency edges for the LFP band.
    bands : dict or None
        Frequency bands for coherency.  Defaults to COHERENCE_BANDS.
    max_duration : float or None
        If set, limit windows to this duration (for combined RMS).
    min_coherency_windows : int
        Minimum number of windows for coherency estimation.
    """
    recording = block.recording
    lfp_recording = block.lfp_recording
    if bands is None:
        bands = COHERENCE_BANDS

    fs = recording.get_sampling_frequency()
    n_samples_total = recording.get_num_samples()
    duration_total = n_samples_total / fs

    # Adaptive interval: ensure enough windows for coherency
    interval = max(
        window_duration,
        min(window_interval, duration_total / min_coherency_windows),
    )

    window_starts_sec = np.arange(
        0, duration_total - window_duration, interval
    )
    n_windows = len(window_starts_sec)
    if n_windows == 0:
        n_windows = 1
        window_starts_sec = np.array([0.0])
    window_samples = int(window_duration * fs)
    window_starts = (window_starts_sec * fs).astype(int)
    timestamps = window_starts_sec + window_duration / 2.0

    # Determine LFP source and its sample rate
    lfp_src = lfp_recording if lfp_recording is not None else recording
    fs_lfp = lfp_src.get_sampling_frequency()

    # LFP spectral parameters (direct FFT, no decimation)
    lfp_window_samples = int(window_duration * fs_lfp)
    lfp_freqs = scipy.fft.rfftfreq(lfp_window_samples, d=1.0 / fs_lfp)

    # AP RMS setup
    if lfp_recording is not None:
        # 1.0: separate short FFT on AP recording
        ap_samples = int(ap_rms_duration * fs)
        ap_freqs = scipy.fft.rfftfreq(ap_samples, d=1.0 / fs)
        ap_mask = ap_freqs > ap_freq_min
    else:
        # 2.0: AP RMS extracted from same wideband FFT as LFP
        ap_mask = lfp_freqs > ap_freq_min
    lfp_mask = (lfp_freqs >= lfp_freq_range[0]) & (
        lfp_freqs <= lfp_freq_range[1]
    )
    n_lfp_psd_bins = int(lfp_mask.sum())

    # Build band masks
    band_masks = {}
    for band_name, (lo, hi) in bands.items():
        band_masks[band_name] = (lfp_freqs >= lo) & (lfp_freqs < hi)

    # Hann window (precomputed, shared across all windows)
    hann = np.hanning(lfp_window_samples).astype(np.float32)
    window_power = np.sum(hann**2)  # for PSD normalization
    # RMS correction: undo the power loss from windowing
    rms_correction = np.sqrt(lfp_window_samples / window_power)

    # Channel group / shank info (from the LFP source)
    try:
        channel_groups = lfp_src.get_property("group")
    except Exception:
        channel_groups = np.zeros(lfp_src.get_num_channels(), dtype=int)
    channel_locations = lfp_src.get_channel_locations()
    unique_groups = np.unique(channel_groups)
    shank_info = {}
    for group_val in unique_groups:
        ch_mask = channel_groups == group_val
        ch_indices = np.where(ch_mask)[0]
        locs = channel_locations[ch_indices]
        depth_order = np.argsort(locs[:, 1])
        shank_info[group_val] = {
            "indices": ch_indices,
            "depth_order": depth_order,
        }

    n_ap_channels = recording.get_num_channels()
    n_lfp_channels = lfp_src.get_num_channels()

    # Output arrays
    rms_ap = np.zeros((n_windows, n_ap_channels), dtype=np.float32)
    rms_lfp = np.zeros((n_windows, n_lfp_channels), dtype=np.float32)

    # Accumulators for PSD and cross-spectral density
    Sij = {
        band_name: np.zeros(
            (n_lfp_channels, n_lfp_channels), dtype=np.complex128
        )
        for band_name in bands
    }
    psd_accum = np.zeros((n_lfp_psd_bins, n_lfp_channels), dtype=np.float64)

    # --- Main loop: submit all reads, process as completed ---
    n_read_workers = min(8, n_windows)

    def _read_window(idx: int) -> tuple[int, np.ndarray, np.ndarray | None]:
        """Read AP (or wideband) + LFP windows for one index."""
        s = int(window_starts[idx])
        ap_data = recording.get_traces(
            start_frame=s,
            end_frame=s + window_samples,
            return_in_uV=True,
        )
        lfp_data = None
        if lfp_recording is not None:
            s_lfp = int(window_starts_sec[idx] * fs_lfp)
            e_lfp = s_lfp + lfp_window_samples
            if e_lfp > lfp_src.get_num_samples():
                e_lfp = lfp_src.get_num_samples()
                s_lfp = max(0, e_lfp - lfp_window_samples)
            lfp_data = lfp_src.get_traces(
                start_frame=s_lfp,
                end_frame=e_lfp,
                return_in_uV=True,
            )
        return idx, ap_data, lfp_data

    # Bounded prefetch: keep n_read_workers reads in flight at a time
    # to avoid buffering all windows in memory simultaneously.
    with ThreadPoolExecutor(max_workers=n_read_workers) as pool:
        pending: dict[object, int] = {}
        next_submit = 0

        def _submit_up_to(limit: int) -> None:
            """Submit reads until *limit* are in flight."""
            nonlocal next_submit
            while next_submit < n_windows and len(pending) < limit:
                f = pool.submit(_read_window, next_submit)
                pending[f] = next_submit
                next_submit += 1

        _submit_up_to(n_read_workers)

        while pending:
            done = next(as_completed(pending))
            del pending[done]
            _submit_up_to(n_read_workers)

            i, raw_data, lfp_data = done.result()
            raw = raw_data.astype(np.float32)

            if lfp_data is not None:
                # --- 1.0: separate AP and LFP recordings ---
                X_ap = scipy.fft.rfft(raw[:ap_samples], axis=0)
                rms_ap[i] = _parseval_rms(X_ap, ap_mask, ap_samples)
                lfp_raw = lfp_data.astype(np.float32)
            else:
                # --- 2.0: wideband ---
                lfp_raw = raw

            # CMR + demean + Hann window -> single FFT
            np.subtract(
                lfp_raw,
                np.median(lfp_raw, axis=1, keepdims=True),
                out=lfp_raw,
            )
            lfp_seg = lfp_raw[: hann.shape[0]]
            np.subtract(lfp_seg, lfp_seg.mean(axis=0), out=lfp_seg)
            np.multiply(lfp_seg, hann[:, None], out=lfp_seg)
            X = scipy.fft.rfft(lfp_seg, axis=0)

            # RMS via Parseval (corrected for Hann window)
            rms_lfp[i] = (
                _parseval_rms(X, lfp_mask, hann.shape[0]) * rms_correction
            )
            if lfp_recording is None:
                rms_ap[i] = (
                    _parseval_rms(X, ap_mask, hann.shape[0]) * rms_correction
                )

            # PSD accumulation (LFP range only)
            psd_accum += np.abs(X[lfp_mask]) ** 2

            # Cross-spectral density per band
            for band_name, mask in band_masks.items():
                X_band = X[mask]
                Sij[band_name] += X_band.conj().T @ X_band

    psd_power, psd_freqs, correlation, complex_coherency = (
        _normalize_spectral_estimates(
            Sij,
            psd_accum,
            shank_info,
            n_windows,
            window_power,
            fs_lfp,
            lfp_mask,
            lfp_freqs,
            bands,
        )
    )

    # Build shank channel metadata from the already-computed shank_info
    shank_channels = []
    for group_val, info in shank_info.items():
        idx = info["indices"]
        order = info["depth_order"]
        shank_channels.append(
            ShankChannels(
                shank_index=int(group_val) + 1,
                locations=channel_locations[idx[order]].copy(),
            )
        )

    # Convert from µV to V (GUI expects volts, multiplies by 1e6)
    uV_to_V = np.float32(1e-6)

    return BlockMetrics(
        block=block,
        rms_ap=rms_ap * uV_to_V,
        rms_lfp=rms_lfp * uV_to_V,
        timestamps=timestamps.astype(np.float64),
        correlation=correlation,
        coherency=complex_coherency,
        psd_power=psd_power * uV_to_V**2,  # PSD is power (V²/Hz)
        psd_freqs=psd_freqs,
        shank_channels=shank_channels,
    )


def _build_channel_maps(
    results: list[BlockMetrics],
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """Build channel index maps for combining blocks with overlapping channels.

    Returns
    -------
    unique_locs : np.ndarray
        Unique channel locations, shape ``(n_unique, 2)``.
    block_channel_maps : list[np.ndarray]
        Per-block arrays mapping each block channel to its index in
        ``unique_locs``.
    total_n_channels : int
        Number of unique channels.
    """
    all_locs = []
    for r in results:
        all_locs.append(r.block.recording.get_channel_locations())
    concat_locs = np.concatenate(all_locs, axis=0)

    _, unique_idx = np.unique(concat_locs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    unique_locs = concat_locs[unique_idx]
    total_n_channels = len(unique_locs)

    block_channel_maps = []
    for locs in all_locs:
        ch_map = np.empty(len(locs), dtype=int)
        for i, loc in enumerate(locs):
            matches = np.where(
                np.all(np.abs(unique_locs - loc) < 1e-6, axis=1)
            )[0]
            ch_map[i] = matches[0] if len(matches) > 0 else i
        block_channel_maps.append(ch_map)

    return unique_locs, block_channel_maps, total_n_channels


def _build_shank_channel_maps(
    results: list[BlockMetrics],
    shank_index: int,
) -> tuple[list[np.ndarray | None], int]:
    """Build per-shank channel maps for assembling coherence matrices.

    Uses the ``ShankChannels`` metadata already stored on each
    ``BlockMetrics`` (computed during ``_compute_all_metrics``),
    avoiding any re-reading from recordings.

    Returns
    -------
    shank_maps : list[np.ndarray | None]
        Per-block arrays mapping each block's shank-channel row/col
        to its index in the unified shank channel set.  ``None`` if
        the block has no channels on this shank.
    n_unique : int
        Total unique channels on this shank across all blocks.
    """
    # Collect per-block shank locations from stored metadata
    per_block_locs: list[np.ndarray | None] = []
    for r in results:
        match = [
            sc for sc in r.shank_channels if sc.shank_index == shank_index
        ]
        per_block_locs.append(match[0].locations if match else None)

    # Build unified unique locations for this shank
    valid_locs = [loc for loc in per_block_locs if loc is not None]
    if not valid_locs:
        return [None] * len(results), 0
    concat = np.concatenate(valid_locs, axis=0)
    _, unique_idx = np.unique(concat, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    unique_shank_locs = concat[unique_idx]
    n_unique = len(unique_shank_locs)

    # Map each block's shank channels to unified indices
    shank_maps: list[np.ndarray | None] = []
    for locs in per_block_locs:
        if locs is None:
            shank_maps.append(None)
            continue
        ch_map = np.empty(len(locs), dtype=int)
        for i, loc in enumerate(locs):
            matches = np.where(
                np.all(np.abs(unique_shank_locs - loc) < 1e-6, axis=1)
            )[0]
            ch_map[i] = matches[0]
        shank_maps.append(ch_map)

    return shank_maps, n_unique


def _assemble_blockwise_coherence(
    results: list[BlockMetrics],
    main_recording_min_secs: float = 600.0,
) -> dict:
    """Assemble per-block coherence into block-diagonal matrices.

    Parameters
    ----------
    results : list[BlockMetrics]
        Per-block results from ``_compute_all_metrics``.
    main_recording_min_secs : float
        Duration threshold to classify blocks as main vs surface.

    Returns
    -------
    dict
        ``correlation``, ``coherency``, ``psd_power``, ``psd_freqs``,
        ``channel_blocks``
    """
    unique_locs, block_channel_maps, total_n_channels = _build_channel_maps(
        results
    )

    # Find all band/shank keys and unique shank indices
    all_keys = set()
    all_shanks = set()
    for result in results:
        all_keys.update(result.correlation.keys())
        all_shanks.update(s for _, s in result.correlation.keys())

    # Build per-shank channel maps once
    shank_maps_cache = {}
    for shank_idx in all_shanks:
        shank_maps_cache[shank_idx] = _build_shank_channel_maps(
            results, shank_idx
        )

    # Assemble per-shank with n_windows-weighted averaging for overlaps
    combined_correlation = {}
    combined_coherency = {}
    for key in all_keys:
        band_name, shank_idx = key
        shank_maps, n_shank_channels = shank_maps_cache[shank_idx]
        if n_shank_channels == 0:
            continue

        corr_mat = np.zeros(
            (n_shank_channels, n_shank_channels), dtype=np.float32
        )
        coh_mat = np.zeros(
            (n_shank_channels, n_shank_channels), dtype=np.complex64
        )
        weight = np.zeros(
            (n_shank_channels, n_shank_channels), dtype=np.float32
        )
        for s_map, result in zip(shank_maps, results):
            if s_map is None or key not in result.correlation:
                continue
            n_win = result.rms_ap.shape[0]
            block_corr = result.correlation[key]
            block_coh = result.coherency[key]
            corr_mat[np.ix_(s_map, s_map)] += block_corr * n_win
            coh_mat[np.ix_(s_map, s_map)] += block_coh * n_win
            weight[np.ix_(s_map, s_map)] += n_win
        mask = weight > 0
        corr_mat[mask] /= weight[mask]
        coh_mat[mask] /= weight[mask]
        combined_correlation[key] = corr_mat
        combined_coherency[key] = coh_mat

    # PSD: assemble with n_windows-weighted averaging for overlaps
    n_lfp_freqs = results[0].psd_power.shape[0]
    combined_psd = np.zeros((n_lfp_freqs, total_n_channels), dtype=np.float32)
    psd_weight = np.zeros(total_n_channels, dtype=np.float32)
    for ch_map, result in zip(block_channel_maps, results):
        n_win = result.rms_ap.shape[0]
        n_ch = result.psd_power.shape[1]
        idx = ch_map[:n_ch]
        combined_psd[:, idx] += result.psd_power * n_win
        psd_weight[idx] += n_win
    psd_mask = psd_weight > 0
    combined_psd[:, psd_mask] /= psd_weight[psd_mask]

    # Channel blocks metadata
    channel_blocks_meta = []
    for block_idx, (r, ch_map) in enumerate(zip(results, block_channel_maps)):
        rec = r.block.recording
        label = (
            "main"
            if rec.get_duration() >= main_recording_min_secs
            else "surface"
        )
        channel_blocks_meta.append(
            {
                "block_index": block_idx,
                "channel_indices": ch_map.tolist(),
                "label": label,
                "n_channels": rec.get_num_channels(),
                "duration_s": float(rec.get_duration()),
            }
        )

    return {
        "correlation": combined_correlation,
        "coherency": combined_coherency,
        "psd_power": combined_psd,
        "psd_freqs": results[0].psd_freqs,
        "channel_blocks": channel_blocks_meta,
    }

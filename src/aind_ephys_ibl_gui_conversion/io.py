"""I/O helpers for loading recordings and saving ephys metrics."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np
import spikeinterface as si

from aind_ephys_ibl_gui_conversion.metrics import (
    _assemble_blockwise_coherence,
    _build_channel_maps,
    _compute_all_metrics,
)
from aind_ephys_ibl_gui_conversion.recording_utils import (
    _stream_to_probe_name,
    get_largest_segment_recordings,
)
from aind_ephys_ibl_gui_conversion.types import (
    BlockMetrics,
    ExperimentBlock,
    ProbeStream,
)

# ---------------------------------------------------------------------------
# Recording loading
# ---------------------------------------------------------------------------

_LoadItem = tuple[str, int, bool]
_LoadedItem = tuple[_LoadItem, si.BaseRecording | None, np.ndarray | None]


def _filter_ap_streams(
    neuropix_streams: list[str],
    stream_to_use: str | None,
) -> list[str]:
    """Return streams that should be loaded as AP/wideband recordings."""
    return [
        stream
        for stream in neuropix_streams
        if "LFP" not in stream
        and (stream_to_use is None or stream == stream_to_use)
    ]


def _build_load_items(
    ap_streams: list[str],
    num_blocks: int,
) -> list[_LoadItem]:
    """Build AP/wideband and optional LFP load requests."""
    load_items: list[_LoadItem] = []
    for stream_name in ap_streams:
        is_1_0 = "AP" in stream_name
        for block_index in range(num_blocks):
            load_items.append((stream_name, block_index, False))
            if is_1_0:
                load_items.append((stream_name, block_index, True))
    return load_items


def _zarr_path_for_item(
    ecephys_compressed_folder: Path,
    item: _LoadItem,
) -> Path:
    """Return the zarr path for one load request."""
    stream_name, block_index, is_lfp = item
    if is_lfp:
        stream_name = stream_name.replace("AP", "LFP")
    return (
        ecephys_compressed_folder
        / f"experiment{block_index + 1}_{stream_name}.zarr"
    )


def _read_contact_ids(zarr_path: Path) -> np.ndarray | None:
    """Read probe contact IDs directly from zarr metadata."""
    zattrs_path = zarr_path / ".zattrs"
    if not zattrs_path.exists():
        return None

    zattrs = json.loads(zattrs_path.read_text())
    raw_ids = (
        zattrs.get("probe", {}).get("probes", [{}])[0].get("contact_ids", [])
    )
    if not raw_ids:
        return None

    return np.array([int(cid.lstrip("e")) for cid in raw_ids])


def _load_one_recording(
    ecephys_compressed_folder: Path,
    item: _LoadItem,
) -> _LoadedItem:
    """Load one zarr recording's metadata and contact IDs."""
    _, _, is_lfp = item
    zarr_path = _zarr_path_for_item(ecephys_compressed_folder, item)
    if is_lfp and not zarr_path.exists():
        return item, None, None

    try:
        rec = si.read_zarr(zarr_path)
    except Exception:
        logging.exception(f"[FFT] Failed to load zarr: {zarr_path}")
        raise
    rec.reset_times()

    contact_ids = None if is_lfp else _read_contact_ids(zarr_path)
    return item, rec, contact_ids


def _load_zarr_metadata(
    load_items: list[_LoadItem],
    ecephys_compressed_folder: Path,
) -> tuple[
    dict[_LoadItem, si.BaseRecording | None],
    dict[_LoadItem, np.ndarray],
]:
    """Load all requested zarr metadata in parallel."""
    loaded = {}
    contact_ids_map = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        args = [ecephys_compressed_folder] * len(load_items)
        for item, rec, contact_ids in pool.map(
            _load_one_recording,
            args,
            load_items,
        ):
            loaded[item] = rec
            if contact_ids is not None:
                contact_ids_map[item] = contact_ids
    return loaded, contact_ids_map


def _log_loaded_recording(
    stream_name: str,
    block_index: int,
    rec: si.BaseRecording,
) -> None:
    """Log AP/wideband recording metadata."""
    logging.info(
        f"[FFT] Loaded stream {stream_name} "
        f"block {block_index}: "
        f"{rec.get_duration():.1f}s, "
        f"{rec.get_num_channels()} channels"
    )


def _log_loaded_lfp_recording(
    stream_name: str,
    block_index: int,
    lfp_rec: si.BaseRecording,
) -> None:
    """Log LFP recording metadata."""
    logging.info(
        f"[FFT] Loaded 1.0 LFP stream "
        f"{stream_name.replace('AP', 'LFP')} "
        f"block {block_index}: "
        f"{lfp_rec.get_sampling_frequency():.0f} Hz, "
        f"{lfp_rec.get_num_channels()} channels"
    )


def _build_experiment_block(
    stream_name: str,
    block_index: int,
    is_1_0: bool,
    loaded: dict[_LoadItem, si.BaseRecording | None],
    contact_ids_map: dict[_LoadItem, np.ndarray],
) -> ExperimentBlock:
    """Build one ExperimentBlock from loaded recordings."""
    ap_item = (stream_name, block_index, False)
    rec = cast(si.BaseRecording, loaded[ap_item])

    _log_loaded_recording(stream_name, block_index, rec)

    lfp_rec = None
    if is_1_0:
        lfp_rec = loaded.get((stream_name, block_index, True))
        if lfp_rec is not None:
            _log_loaded_lfp_recording(stream_name, block_index, lfp_rec)

    contact_ids = contact_ids_map.get(ap_item)
    if contact_ids is None:
        contact_ids = np.arange(rec.get_num_channels())

    return ExperimentBlock(
        recording=rec,
        lfp_recording=lfp_rec,
        block_index=block_index,
        contact_ids=contact_ids,
    )


def _build_probe_stream(
    stream_name: str,
    num_blocks: int,
    results_folder: Path,
    loaded: dict[_LoadItem, si.BaseRecording | None],
    contact_ids_map: dict[_LoadItem, np.ndarray],
) -> ProbeStream:
    """Build one ProbeStream from loaded block recordings."""
    is_1_0 = "AP" in stream_name
    probe_name = _stream_to_probe_name(stream_name)
    blocks = [
        _build_experiment_block(
            stream_name,
            block_index,
            is_1_0,
            loaded,
            contact_ids_map,
        )
        for block_index in range(num_blocks)
    ]

    return ProbeStream(
        stream_name=stream_name,
        probe_name=probe_name,
        blocks=blocks,
        output_folder=results_folder / probe_name,
    )


def load_probe_streams(
    neuropix_streams: list[str],
    num_blocks: int,
    ecephys_compressed_folder: Path,
    results_folder: Path,
    stream_to_use: str | None = None,
    main_recording_min_secs: int = 600,
) -> list[ProbeStream]:
    """Load raw (unfiltered) recordings into ProbeStream objects.

    For Neuropixels 1.0 probes (stream name contains "AP"), also
    loads the corresponding LFP stream.

    Parameters
    ----------
    neuropix_streams : list[str]
        Stream names (LFP streams are skipped).
    num_blocks : int
        Number of experiment blocks.
    ecephys_compressed_folder : Path
        Path to compressed zarr recordings.
    results_folder : Path
        Base path for output folders.
    stream_to_use : str or None
        If provided, only process this stream.
    main_recording_min_secs : int
        Minimum duration to classify as main (vs surface).

    Returns
    -------
    list[ProbeStream]
        One ProbeStream per stream, each containing ExperimentBlocks.
    """
    ap_streams = _filter_ap_streams(neuropix_streams, stream_to_use)
    load_items = _build_load_items(ap_streams, num_blocks)
    loaded, contact_ids_map = _load_zarr_metadata(
        load_items,
        ecephys_compressed_folder,
    )
    # Filter to streams we care about
    ap_streams = [
        s
        for s in neuropix_streams
        if "LFP" not in s and (stream_to_use is None or s == stream_to_use)
    ]

    # Build list of all zarr paths to load in parallel
    load_items = []
    for stream_name in ap_streams:
        is_1_0 = "AP" in stream_name
        for block_index in range(num_blocks):
            load_items.append(
                (stream_name, block_index, False)  # AP/wideband
            )
            if is_1_0:
                load_items.append(
                    (stream_name, block_index, True)  # LFP
                )

    def _load_one(item):
        """Load one zarr recording's metadata."""
        stream_name, block_index, is_lfp = item
        if is_lfp:
            lfp_stream_name = stream_name.replace("AP", "LFP")
            zarr_path = (
                ecephys_compressed_folder
                / f"experiment{block_index + 1}_{lfp_stream_name}.zarr"
            )
        else:
            zarr_path = (
                ecephys_compressed_folder
                / f"experiment{block_index + 1}_{stream_name}.zarr"
            )
        if is_lfp and not zarr_path.exists():
            return item, None
        try:
            rec = si.read_zarr(zarr_path)
        except Exception:
            logging.exception(f"[FFT] Failed to load zarr: {zarr_path}")
            raise
        # Multi-segment recordings (e.g. an experiment with multiple
        # acquisition starts/stops) must be reduced to a single segment
        # before any segment-agnostic call like get_duration(). For
        # alignment we keep the largest segment.
        if rec.get_num_segments() > 1:
            (rec,) = get_largest_segment_recordings([rec])
            logging.info(
                f"[FFT] Multi-segment recording {zarr_path.name} "
                "reduced to largest segment"
            )
        rec.reset_times()
        return item, rec

    # Load all zarr metadata in parallel
    loaded = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for item, rec in pool.map(_load_one, load_items):
            loaded[item] = rec

    # Build ProbeStream objects
    results_folder = Path(results_folder)
    return [
        _build_probe_stream(
            stream_name,
            num_blocks,
            results_folder,
            loaded,
            contact_ids_map,
        )
        for stream_name in ap_streams
    ]


# ---------------------------------------------------------------------------
# Output save helpers
# ---------------------------------------------------------------------------


def _save_rms(
    output_folder: Path,
    result: BlockMetrics,
    tag: str,
) -> None:
    """Save RMS time series for a single block."""
    np.save(
        output_folder / f"_iblqc_ephysTimeRmsAP{tag}.rms.npy",
        result.rms_ap,
    )
    np.save(
        output_folder / f"_iblqc_ephysTimeRmsAP{tag}.timestamps.npy",
        result.timestamps,
    )
    np.save(
        output_folder / f"_iblqc_ephysTimeRmsLF{tag}.rms.npy",
        result.rms_lfp,
    )
    np.save(
        output_folder / f"_iblqc_ephysTimeRmsLF{tag}.timestamps.npy",
        result.timestamps,
    )


def _save_combined_rms_summary(
    output_folder: Path,
    results: list[BlockMetrics],
) -> None:
    """Save mean RMS per channel across all blocks.

    Produces a single value per unique channel position (no time
    axis).  Overlapping channels between blocks are combined by
    averaging power (RMS²), then taking the square root — the
    mathematically correct way to combine RMS estimates.
    """
    _, block_maps, n_unique = _build_channel_maps(results)

    ap_combined = np.zeros(n_unique, dtype=np.float64)
    lfp_combined = np.zeros(n_unique, dtype=np.float64)
    weight_sum = np.zeros(n_unique, dtype=np.float64)

    for r, ch_map in zip(results, block_maps):
        n_win = r.rms_ap.shape[0]
        # Per-block mean power (RMS² averaged over time), weighted
        # by n_windows to minimize variance of the combined estimate
        ap_power = np.mean(r.rms_ap**2, axis=0)
        lfp_power = np.mean(r.rms_lfp**2, axis=0)
        ap_combined[ch_map] += ap_power * n_win
        lfp_combined[ch_map] += lfp_power * n_win
        weight_sum[ch_map] += n_win

    # Weighted average power, then sqrt → RMS
    mask = weight_sum > 0
    ap_combined[mask] /= weight_sum[mask]
    lfp_combined[mask] /= weight_sum[mask]
    ap_rms = np.sqrt(ap_combined).astype(np.float32)
    lfp_rms = np.sqrt(lfp_combined).astype(np.float32)

    np.save(
        output_folder / "_iblqc_ephysTimeRmsAP.rms.npy",
        ap_rms[np.newaxis, :],
    )
    np.save(
        output_folder / "_iblqc_ephysTimeRmsAP.timestamps.npy",
        np.array([0.0]),
    )
    np.save(
        output_folder / "_iblqc_ephysTimeRmsLF.rms.npy",
        lfp_rms[np.newaxis, :],
    )
    np.save(
        output_folder / "_iblqc_ephysTimeRmsLF.timestamps.npy",
        np.array([0.0]),
    )


def _save_spectral_outputs(
    output_folder: Path,
    results: list[BlockMetrics],
) -> None:
    """Save PSD, correlation, and coherency from one or more blocks.

    When ``len(results) > 1``, assembles block-diagonal coherence
    via ``_assemble_blockwise_coherence`` and writes
    ``channel_blocks.json``.  When ``len(results) == 1``, saves
    directly from the single result (no assembly needed).
    """
    if len(results) > 1:
        assembled = _assemble_blockwise_coherence(results)
        psd_power = assembled["psd_power"]
        psd_freqs = assembled["psd_freqs"]
        correlation = assembled["correlation"]
        coherency = assembled["coherency"]
        channel_blocks = assembled["channel_blocks"]
    else:
        r = results[0]
        psd_power = r.psd_power
        psd_freqs = r.psd_freqs
        correlation = r.correlation
        coherency = r.coherency
        channel_blocks = None

    np.save(
        output_folder / "_iblqc_ephysSpectralDensityLF.power.npy",
        psd_power,
    )
    np.save(
        output_folder / "_iblqc_ephysSpectralDensityLF.freqs.npy",
        psd_freqs,
    )

    band_corr_folder = output_folder / "band_corr"
    band_corr_folder.mkdir(exist_ok=True)
    for (band_name, shank_idx), corr_mat in correlation.items():
        np.save(
            band_corr_folder / f"{band_name}_shank{shank_idx}_mean_corr.npy",
            corr_mat,
        )
    for (band_name, shank_idx), coh_mat in coherency.items():
        np.save(
            band_corr_folder / f"{band_name}_shank{shank_idx}_coherency.npy",
            coh_mat,
        )

    if channel_blocks is not None:
        with open(band_corr_folder / "channel_blocks.json", "w") as f:
            json.dump({"blocks": channel_blocks}, f, indent=2)


def _save_method_metadata(
    output_folder: Path,
    rms_window_interval: float,
    rms_window_duration: float,
) -> None:
    """Save method metadata for forward compatibility.

    The GUI and preprocessing manifest can use this to
    determine what kind of metrics are in the output files
    (e.g. coherence vs correlation).
    """
    metadata = {
        "method": "fft_coherence",
        "version": "1.0",
        "metrics": {
            "rms": "parseval_fft",
            "band_corr": "magnitude_squared_coherence",
            "psd": "welch_fft",
        },
        "parameters": {
            "rms_window_interval_s": rms_window_interval,
            "rms_window_duration_s": rms_window_duration,
            "cmr": True,
        },
    }
    path = output_folder / "_iblqc_metrics.method.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def _save_channel_metadata(
    output_folder: Path,
    blocks: list[ExperimentBlock],
) -> None:
    """Save deduplicated channel locations and indices.

    Uses contact_ids for deduplication and (x, y) positions for
    output ordering (by depth then lateral position).
    """
    all_ids = np.concatenate([b.contact_ids for b in blocks])
    all_locs = np.concatenate(
        [b.recording.get_channel_locations() for b in blocks], axis=0
    )

    # Deduplicate by contact_id
    _, first_idx = np.unique(all_ids, return_index=True)
    unique_locs = all_locs[first_idx]

    # Order by depth (y), then lateral position (x)
    output_order = np.lexsort((unique_locs[:, 0], unique_locs[:, 1]))
    unique_locs = unique_locs[output_order]

    np.save(output_folder / "channels.localCoordinates.npy", unique_locs)
    np.save(
        output_folder / "channels.rawInd.npy",
        np.arange(unique_locs.shape[0]),
    )


def _assemble_and_save_stream(
    stream: ProbeStream,
    results: list[BlockMetrics],
    rms_window_interval: float,
    rms_window_duration: float,
) -> None:
    """Assemble block results and save all outputs for one stream.

    Handles partitioning main/surface results, saving combined and
    main RMS, spectral outputs, channel metadata, and method metadata.

    Parameters
    ----------
    stream : ProbeStream
        Probe stream with output folder and block info.
    results : list[BlockMetrics]
        Per-block metric results, sorted by block index.
    rms_window_interval : float
        Window interval parameter (saved to method metadata).
    rms_window_duration : float
        Window duration parameter (saved to method metadata).
    """
    output_folder = stream.output_folder
    output_folder.mkdir(exist_ok=True)

    # Find the main block result (longest recording)
    main_result = max(
        results,
        key=lambda r: r.block.duration,
    )

    if stream.has_surface:
        # Combined: probe summary RMS + coherence + channel metadata
        _save_combined_rms_summary(output_folder, results)
        _save_spectral_outputs(output_folder, results)
        _save_channel_metadata(output_folder, stream.blocks)

        # Main: full RMS time series (only main block channels)
        _save_rms(output_folder, main_result, tag="Main")
    else:
        # No surface: single block, save everything
        _save_rms(output_folder, main_result, tag="")
        _save_spectral_outputs(output_folder, [main_result])
        _save_channel_metadata(output_folder, [main_result.block])

    _save_method_metadata(
        output_folder,
        rms_window_interval=rms_window_interval,
        rms_window_duration=rms_window_duration,
    )


def process_stream_fft(
    block: ExperimentBlock,
    output_folder: Path,
    compute_coherence: bool = True,
    save_channel_metadata: bool = True,
    rms_window_interval: float = 30.0,
    rms_window_duration: float = 4.0,
    tag: str = "",
    **kwargs,
):
    """Compute all ephys QC metrics for one recording.

    Convenience wrapper around ``_compute_all_metrics`` that
    saves output files.  Used by tests; the main code path
    goes through ``_process_block`` in ``extract_continuous``.
    """
    output_folder.mkdir(exist_ok=True)

    metrics = _compute_all_metrics(
        block,
        window_interval=rms_window_interval,
        window_duration=rms_window_duration,
    )

    np.save(
        output_folder / f"_iblqc_ephysTimeRmsAP{tag}.rms.npy",
        metrics.rms_ap,
    )
    np.save(
        output_folder / f"_iblqc_ephysTimeRmsAP{tag}.timestamps.npy",
        metrics.timestamps,
    )
    np.save(
        output_folder / f"_iblqc_ephysTimeRmsLF{tag}.rms.npy",
        metrics.rms_lfp,
    )
    np.save(
        output_folder / f"_iblqc_ephysTimeRmsLF{tag}.timestamps.npy",
        metrics.timestamps,
    )

    if compute_coherence:
        np.save(
            output_folder / f"_iblqc_ephysSpectralDensityLF{tag}.power.npy",
            metrics.psd_power,
        )
        np.save(
            output_folder / f"_iblqc_ephysSpectralDensityLF{tag}.freqs.npy",
            metrics.psd_freqs,
        )
        band_corr_folder = output_folder / "band_corr"
        band_corr_folder.mkdir(exist_ok=True)
        for (band_name, shank_idx), corr in metrics.correlation.items():
            np.save(
                band_corr_folder
                / f"{band_name}_shank{shank_idx}_mean_corr.npy",
                corr,
            )
        for (band_name, shank_idx), coh in metrics.coherency.items():
            np.save(
                band_corr_folder
                / f"{band_name}_shank{shank_idx}_coherency.npy",
                coh,
            )

    if save_channel_metadata:
        _save_channel_metadata(output_folder, [block])

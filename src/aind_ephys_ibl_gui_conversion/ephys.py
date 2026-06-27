"""
Functions to process ephys data
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import scipy.fft

from aind_ephys_ibl_gui_conversion.io import (
    _assemble_and_save_stream,
    _save_channel_metadata,
    _save_method_metadata,
    _save_rms,
    _save_spectral_outputs,
    load_probe_streams,
    process_stream_fft,
)
from aind_ephys_ibl_gui_conversion.metrics import (
    COHERENCE_BANDS,
    _assemble_blockwise_coherence,
    _compute_all_metrics,
    _parseval_rms,
)
from aind_ephys_ibl_gui_conversion.recording_utils import (
    _merge_separate_asset_recording_dicts,
    _stream_to_probe_name,
    get_ecephys_stream_names,
    get_largest_segment_recordings,
    get_main_recording_from_list,
    merge_probe_streams,
)
from aind_ephys_ibl_gui_conversion.spikes import extract_spikes
from aind_ephys_ibl_gui_conversion.types import (
    BlockMetrics,
    ExperimentBlock,
    ProbeStream,
)

# Re-export for backward API compatibility
__all__ = [
    "extract_continuous",
    "extract_spikes",
    # re-exported from recording_utils
    "_merge_separate_asset_recording_dicts",
    "_stream_to_probe_name",
    "get_ecephys_stream_names",
    "get_largest_segment_recordings",
    "get_main_recording_from_list",
    "merge_probe_streams",
    # re-exported from metrics
    "COHERENCE_BANDS",
    "_parseval_rms",
    "_compute_all_metrics",
    "_assemble_blockwise_coherence",
    # re-exported from io
    "load_probe_streams",
    "_save_rms",
    "_save_spectral_outputs",
    "_save_method_metadata",
    "_save_channel_metadata",
    "_assemble_and_save_stream",
    "process_stream_fft",
    # re-exported from types
    "BlockMetrics",
    "ExperimentBlock",
    "ProbeStream",
]


def extract_continuous(
    sorting_folder: Path,
    results_folder: Path,
    stream_to_use: str | None = None,
    main_recording_min_secs: int = 600,
    probe_surface_finding: Path | None = None,
    rms_window_interval: float = 30.0,
    rms_window_duration: float = 4.0,
    num_parallel_jobs: int = 4,
    # Legacy parameters kept for API compatibility (ignored)
    coherence_duration: float = 60.0,
    coherence_window_duration: float = 2.0,
    lfp_freq_min: float = 1,
    lfp_freq_max: float = 300,
    target_sample_rate: float = 1250,
    target_freq_resolution_psd: float = 0.5,
    chunk_duration: float = 15.0,
    lfp_correlation_min_secs: int = 600,
    lfp_correlation_num_bins: int = 5,
    lfp_bandpass_margin_ms: float = 3000.0,
):
    """Extract QC metrics from raw ephys data using FFT-based computation.

    Computes AP and LFP RMS, LFP power spectral density, and
    per-band coherence matrices directly from the raw recording
    via FFT.

    Parameters
    ----------
    sorting_folder : Path
        Path to the sorted data folder.
    results_folder : Path
        Path where output files will be saved.
    stream_to_use : str or None
        If provided, only process this stream.
    main_recording_min_secs : int
        Minimum duration (seconds) to classify as main recording.
    probe_surface_finding : Path or None
        Path to surface-finding data asset, if separate.
    rms_window_interval : float
        Maximum interval in seconds between windows.
    rms_window_duration : float
        Duration of each window in seconds (LFP metrics + RMS).
    num_parallel_jobs : int
        Number of blocks to process in parallel.
    """
    session_folder = Path(str(sorting_folder).split("_sorted")[0])

    # Enable multi-threaded wavpack decompression if available
    try:
        from wavpack_numcodecs import set_num_decoding_threads

        set_num_decoding_threads(num_parallel_jobs)
        logging.info(
            f"[FFT] WavPack decoding threads set to {num_parallel_jobs}"
        )
    except ImportError:
        pass

    neuropix_streams, ecephys_compressed_folder, num_blocks = (
        get_ecephys_stream_names(session_folder)
    )

    if stream_to_use is not None:
        logging.info(
            f"[FFT] Stream filter active: only processing {stream_to_use}"
        )

    # Load raw recordings into ProbeStream objects
    streams = load_probe_streams(
        neuropix_streams,
        num_blocks,
        ecephys_compressed_folder,
        results_folder,
        stream_to_use=stream_to_use,
        main_recording_min_secs=main_recording_min_secs,
    )

    # Handle separate surface-finding asset
    if probe_surface_finding is not None:
        (
            neuropix_streams_surface,
            ecephys_compressed_folder_surface,
            _,
        ) = get_ecephys_stream_names(probe_surface_finding)

        surface_streams = load_probe_streams(
            neuropix_streams_surface,
            num_blocks,
            ecephys_compressed_folder_surface,
            results_folder,
            stream_to_use=stream_to_use,
            main_recording_min_secs=main_recording_min_secs,
        )
        streams = merge_probe_streams(streams, surface_streams)

    # Divide FFT threads across pool workers to avoid oversubscription
    n_cpus = os.cpu_count() or 1
    fft_workers = max(1, n_cpus // max(1, num_parallel_jobs))

    # Count total blocks for logging
    total_blocks = sum(len(s.blocks) for s in streams)
    logging.info(
        f"[FFT] Processing {len(streams)} stream(s), "
        f"{total_blocks} block tasks "
        f"with {num_parallel_jobs} workers"
    )

    def _process_block(block: ExperimentBlock) -> BlockMetrics:
        """Process one block: all metrics from sparse windows."""
        with scipy.fft.set_workers(fft_workers):
            block_start = datetime.now()
            label = f"block[{block.block_index}]"
            logging.info(
                f"[FFT] {label}: starting "
                f"({block.recording.get_duration():.0f}s, "
                f"{block.recording.get_num_channels()} ch)"
            )

            result = _compute_all_metrics(
                block,
                window_interval=rms_window_interval,
                window_duration=rms_window_duration,
            )

            total = (datetime.now() - block_start).total_seconds()
            n_win = len(result.timestamps)
            logging.info(
                f"[FFT] {label}: done ({n_win} windows, {total:.0f}s)"
            )

            return result

    # --- Execute all blocks across all streams ---
    start = datetime.now()

    all_blocks = [
        (stream, block) for stream in streams for block in stream.blocks
    ]

    if num_parallel_jobs <= 1:
        all_results = [
            (stream, _process_block(block)) for stream, block in all_blocks
        ]
    else:
        all_results = []
        with ThreadPoolExecutor(max_workers=num_parallel_jobs) as pool:
            futures = {
                pool.submit(_process_block, block): (stream, block)
                for stream, block in all_blocks
            }
            for future in as_completed(futures):
                stream, block = futures[future]
                try:
                    all_results.append((stream, future.result()))
                except Exception:
                    logging.exception(
                        f"[FFT] Block "
                        f"{stream.stream_name}[{block.block_index}]"
                        f" failed"
                    )
                    raise

    # --- Assemble per stream ---
    for stream in streams:
        # Build a lookup from block identity to result
        block_results = {
            id(r.block): r for s, r in all_results if s is stream
        }
        # Preserve stream.blocks ordering (main first, surface second)
        # to stay consistent with _save_channel_metadata
        results = [block_results[id(b)] for b in stream.blocks]

        _assemble_and_save_stream(
            stream,
            results,
            rms_window_interval=rms_window_interval,
            rms_window_duration=rms_window_duration,
        )

    total_elapsed = (datetime.now() - start).total_seconds()
    logging.info(f"[FFT] All streams complete in {total_elapsed:.1f}s")

import numpy as np
from scipy.signal import fftconvolve, correlate
from typing import Tuple
from ofdm.config import OFDMConfig


def calculate_schmidl_cox_metrics(
    rx_signal: np.ndarray, config: OFDMConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Schmidl-Cox M and P metrics for packet detection and timing synchronization.

    Returns:
        M_metric: Timing metric (peaks at packet start)
        P_metric: Cross-correlation metric
    """
    L = config.N // 2

    r_upper = rx_signal[L:]
    r_lower = rx_signal[:-L]

    P = fftconvolve(np.conj(r_lower) * r_upper, np.ones(L), mode="valid")

    energy = np.abs(r_upper) ** 2
    R = fftconvolve(energy, np.ones(L), mode="valid")

    min_len = min(len(P), len(R))
    P = P[:min_len]
    R = R[:min_len]

    R[R == 0] = 1e-10

    # suppress M in low-energy regions (noise floor) to avoid false peaks
    avg_energy = np.mean(R)
    valid_indices = R > avg_energy * 0.1

    M = np.zeros_like(R)
    M[valid_indices] = (np.abs(P[valid_indices]) ** 2) / (R[valid_indices] ** 2)

    return M, P


def find_start_idx(
    M_metric: np.ndarray,
    config: OFDMConfig,
    rx_signal: np.ndarray,
    known_sync_time: np.ndarray,
    search_window: int,
) -> int:
    """
    Determine the exact sample index where the packet begins (Start of CP).
    Args:
        rx_signal: The raw recieved complex samples
        M_metric: The Schmidl_Cox metric (get from calculate_schmidl_cox_metrics())
        known_sync_time: Ideal time-domain sync symbol (with CP)
        search_window: How many samlpes Left and Right to search during fine sync
    Returns:
        The exact idx of the packet start (beginning of CP)
    """

    # -------- Coarse Sync ---------
    coarse_symbol_start = np.argmax(M_metric)

    # Estimate packet start
    coarse_packet_start = coarse_symbol_start - config.CP_LEN

    # Prevent negative index
    if coarse_packet_start < 0:
        coarse_packet_start = 0

    # -------- Fine Sync ---------
    start_search = max(0, coarse_packet_start - search_window)
    end_search = min(
        len(rx_signal), coarse_packet_start + len(known_sync_time) + search_window
    )

    rx_chunk = rx_signal[start_search:end_search]

    # Cross-Correlate
    corr = correlate(rx_chunk, known_sync_time, mode="valid")

    fine_offset = np.argmax(np.abs(corr))

    # Calculate final index
    exact_start_index = start_search + fine_offset

    return exact_start_index


def estimate_cfo_coarse(P_value: complex, config: OFDMConfig) -> float:
    # Calculate Phase difference with P metrics
    theta = np.angle(P_value)
    cfo_hz = (theta * config.FS) / (np.pi * config.N)
    return cfo_hz

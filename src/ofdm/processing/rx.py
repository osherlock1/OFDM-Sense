import numpy as np
import json
from ofdm.config import OFDMConfig
from ofdm.core import sync, waveform, preamble, payload
from ofdm.channel import CHEST, cfo


def _load_rx_data(rx_path: str, ref_path: str) -> tuple:
    rx_raw = np.fromfile(rx_path, dtype=np.complex64)
    with open(ref_path) as f:
        ref_data = json.load(f)
    return rx_raw, ref_data


def _synchronize(rx_raw: np.ndarray, ref_data: dict, ofdm_conf: OFDMConfig) -> int:
    sync_ref = np.array(ref_data["sync_ref_real"]).astype(complex) + 1j * np.array(
        ref_data["sync_ref_imag"]
    ).astype(complex)
    M, _ = sync.calculate_schmidl_cox_metrics(rx_signal=rx_raw, config=ofdm_conf)
    return sync.find_start_idx(
        M_metric=M,
        config=ofdm_conf,
        rx_signal=rx_raw,
        known_sync_time=sync_ref,
        search_window=500,
    )


def _estimate_and_correct_cfo(
    rx_raw: np.ndarray, ref_data: dict, start_idx: int, ofdm_conf: OFDMConfig
) -> tuple:
    sym_len = ofdm_conf.CP_LEN + ofdm_conf.N
    coarse_pilot_start = start_idx + sym_len
    search_margin = ofdm_conf.CP_LEN
    pilot_search_area = rx_raw[
        coarse_pilot_start - search_margin : coarse_pilot_start
        + sym_len
        + search_margin
    ]

    tx_pilot_ref = np.array(ref_data["pilot_ref_real"]).astype(complex) + 1j * np.array(
        ref_data["pilot_ref_imag"]
    ).astype(complex)

    best_cfo, best_delay_rel, _ = cfo.estimate_cfo(
        tx_ref=tx_pilot_ref,
        rx_signal=pilot_search_area,
        fs=ofdm_conf.FS,
        n_bins=2**14,
    )
    best_cfo *= -1  # sign is inverted empirically

    actual_pilot_start = (coarse_pilot_start - search_margin) + best_delay_rel
    refined_packet_start = actual_pilot_start - sym_len

    correction_vector = np.exp(
        -1j * 2 * np.pi * best_cfo * np.arange(len(rx_raw)) / ofdm_conf.FS
    )
    return rx_raw * correction_vector, refined_packet_start, tx_pilot_ref


def _extract_symbols(
    rx_corrected: np.ndarray,
    refined_packet_start: int,
    n_payload_syms: int,
    ofdm_conf: OFDMConfig,
) -> tuple:
    sym_len = ofdm_conf.N + ofdm_conf.CP_LEN
    total_symbols = 2 + n_payload_syms  # sync + pilot + payload
    total_samples = sym_len * total_symbols

    if refined_packet_start + total_samples > len(rx_corrected):
        refined_packet_start = 0

    packet_time = rx_corrected[
        refined_packet_start : refined_packet_start + total_samples
    ]
    symbols = np.split(packet_time, total_symbols)
    return symbols[1], symbols[2:], refined_packet_start  # pilot, payload, start_idx


def _demodulate_payload(
    rx_pilot_sym: np.ndarray,
    rx_payload_syms: list,
    tx_pilot_ref: np.ndarray,
    ofdm_conf: OFDMConfig,
) -> np.ndarray:
    tx_pilot_freq = waveform.time_to_freq(
        waveform.remove_cp(tx_pilot_ref, cp_len=ofdm_conf.CP_LEN)
    )
    rx_pilot_freq = waveform.time_to_freq(
        waveform.remove_cp(rx_pilot_sym, cp_len=ofdm_conf.CP_LEN)
    )
    Lambda_est = CHEST.channel_estimation_calc(
        rx_pilot_freq=rx_pilot_freq, tx_pilot_ref=tx_pilot_freq, config=ofdm_conf
    )

    pilots_idx = ofdm_conf._idx(np.array(ofdm_conf.pilot_carriers))
    tx_pilot_vals = preamble.generate_pilot_vals(config=ofdm_conf)
    demodulated_data = []

    for sym_time in rx_payload_syms:
        sym_freq = waveform.time_to_freq(
            waveform.remove_cp(sym_time, cp_len=ofdm_conf.CP_LEN)
        )
        sym_eq = CHEST.apply_gains(sym_freq, Lambda_est=Lambda_est)

        phase_drift = np.angle(np.vdot(tx_pilot_vals, sym_eq[pilots_idx]))
        sym_final = sym_eq * np.exp(-1j * phase_drift)

        demodulated_data.extend(payload.extract_data(sym_final, config=ofdm_conf))

    return np.array(demodulated_data) * np.sqrt(10)


def unpack_rx_file(ofdm_conf: OFDMConfig, rx_path: str, ref_path: str) -> tuple:
    """
    Unpack a received OFDM binary file: sync, CFO correction, channel estimation, demodulation.
    Returns demodulated IQ data, reference data dict, and packet start index.
    """
    rx_raw, ref_data = _load_rx_data(rx_path, ref_path)
    start_idx = _synchronize(rx_raw, ref_data, ofdm_conf)
    rx_corrected, refined_start, tx_pilot_ref = _estimate_and_correct_cfo(
        rx_raw, ref_data, start_idx, ofdm_conf
    )
    rx_pilot_sym, rx_payload_syms, refined_start = _extract_symbols(
        rx_corrected, refined_start, ref_data["n_data_symb"], ofdm_conf
    )
    demodulated_data = _demodulate_payload(
        rx_pilot_sym, rx_payload_syms, tx_pilot_ref, ofdm_conf
    )
    return demodulated_data, ref_data, refined_start


def unpack_rx_packet(
    ofdm_conf: OFDMConfig,
    rx_raw: np.ndarray,
    sync_ref_time: np.ndarray,
    tx_pilot_ref: np.ndarray,
    n_data_symbols: int,
) -> tuple:
    """
    Unpack a single OFDM packet already sliced from a raw rx array (rather than a file):
    sync, CFO correction, channel estimation, demodulation.
    Returns demodulated IQ data and the refined packet start index.
    """
    M, _ = sync.calculate_schmidl_cox_metrics(rx_signal=rx_raw, config=ofdm_conf)
    start_idx = sync.find_start_idx(
        M_metric=M,
        config=ofdm_conf,
        rx_signal=rx_raw,
        known_sync_time=sync_ref_time,
        search_window=500,
    )

    sym_len = ofdm_conf.CP_LEN + ofdm_conf.N
    coarse_pilot_start = start_idx + sym_len
    search_margin = ofdm_conf.CP_LEN
    pilot_search_area = rx_raw[
        coarse_pilot_start - search_margin : coarse_pilot_start
        + sym_len
        + search_margin
    ]

    best_cfo, best_delay_rel, _ = cfo.estimate_cfo(
        tx_ref=tx_pilot_ref,
        rx_signal=pilot_search_area,
        fs=ofdm_conf.FS,
        n_bins=2**14,
    )
    best_cfo *= -1

    actual_pilot_start = (coarse_pilot_start - search_margin) + best_delay_rel
    refined_packet_start = actual_pilot_start - sym_len

    correction_vector = np.exp(
        -1j * 2 * np.pi * best_cfo * np.arange(len(rx_raw)) / ofdm_conf.FS
    )
    rx_corrected = rx_raw * correction_vector

    total_symbols = 2 + n_data_symbols
    total_samples = sym_len * total_symbols
    if refined_packet_start + total_samples > len(rx_raw):
        refined_packet_start = 0

    packet_time = rx_corrected[
        refined_packet_start : refined_packet_start + total_samples
    ]
    symbols = np.split(packet_time, total_symbols)
    rx_pilot_sym, rx_payload_syms = symbols[1], symbols[2:]

    tx_pilot_freq = waveform.time_to_freq(
        waveform.remove_cp(tx_pilot_ref, cp_len=ofdm_conf.CP_LEN)
    )
    rx_pilot_freq = waveform.time_to_freq(
        waveform.remove_cp(rx_pilot_sym, cp_len=ofdm_conf.CP_LEN)
    )
    Lambda_est = CHEST.channel_estimation_calc(
        rx_pilot_freq=rx_pilot_freq, tx_pilot_ref=tx_pilot_freq, config=ofdm_conf
    )

    pilots_idx = ofdm_conf._idx(np.array(ofdm_conf.pilot_carriers))
    tx_pilot_vals = preamble.generate_pilot_vals(config=ofdm_conf)
    demodulated_data = []

    for sym_time in rx_payload_syms:
        sym_freq = waveform.time_to_freq(
            waveform.remove_cp(sym_time, cp_len=ofdm_conf.CP_LEN)
        )
        sym_eq = CHEST.apply_gains(sym_freq, Lambda_est=Lambda_est)

        phase_drift = np.angle(np.vdot(tx_pilot_vals, sym_eq[pilots_idx]))
        sym_final = sym_eq * np.exp(-1j * phase_drift)

        demodulated_data.extend(payload.extract_data(sym_final, config=ofdm_conf))

    return np.array(demodulated_data), refined_packet_start


def extract_packet(
    rx_data: np.ndarray, start_idx: int, total_symbols: int, ofdm_conf: OFDMConfig
) -> np.ndarray:
    """
    Slice raw rx_data to return only the packet samples.

    Args:
        rx_data: Received RX IQ data from .dat file
        start_idx: Packet start index, get from unpack_rx_file()
        total_symbols: Total number of sync, pilot, and data symbols
        ofdm_conf: OFDMConfig data class
    """
    sym_len = ofdm_conf.N + ofdm_conf.CP_LEN
    total_samples = sym_len * total_symbols
    start = max(0, int(start_idx))
    end = min(len(rx_data), int(start_idx + total_samples))
    return rx_data[start:end]


def normalize_rx_signal(rx_data: np.ndarray) -> np.ndarray:
    """Normalize raw RX IQ data to +-0.9."""
    max_val = np.max(np.abs(rx_data))
    if max_val == 0:
        raise ValueError(
            "Cannot normalize signal: max amplitude is zero. Check the .dat file for bad capture."
        )
    return rx_data * 0.9 / max_val

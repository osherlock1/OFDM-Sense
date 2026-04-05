import numpy as np
import json
from ofdm.config import OFDMConfig
from ofdm.core import sync, waveform, preamble, payload
from ofdm.channel import CHEST, cfo


def unpack_rx_file(ofdm_conf:OFDMConfig, rx_path:str, ref_path:str, sim:bool = False)->np.ndarray:
    """  
    Take raw ofdm binary data from the rx and unpack it including syncronization, channel estimation, and cfo calibration.
    """

    
    if sim == False:
        rx_raw = np.fromfile(rx_path, dtype=np.complex64)
    else: 
        rx_raw = np.fromfile(rx_path, dtype=np.complex64)
    with open(ref_path) as f:
        ref_data = json.load(f)
    
    #Unpack Referense Sync Symbol
    sync_ref_real = np.array(ref_data['sync_ref_real']).astype(complex)
    sync_ref_imag = np.array(ref_data["sync_ref_imag"]).astype(complex)
    sync_ref_time = sync_ref_real + 1j * sync_ref_imag
    n_payload_syms = ref_data["n_data_symb"]

    # ---------- Syncronization -------------
    M, P = sync.calculate_schmidl_cox_metrics(rx_signal=rx_raw, config=ofdm_conf)
    start_idx = sync.find_start_idx(
        M_metric=M,
        config=ofdm_conf,
        rx_signal=rx_raw,
        known_sync_time=sync_ref_time,
        search_window=500
    )

    # locate pilot symbol
    sync_len = ofdm_conf.CP_LEN + ofdm_conf.N
    pilot_len = ofdm_conf.CP_LEN + ofdm_conf.N
    coarse_pilot_start = start_idx + sync_len

    #Create a search window
    search_margin = ofdm_conf.CP_LEN
    pilot_chunk_start = coarse_pilot_start - search_margin
    pilot_chunk_end = coarse_pilot_start + pilot_len + search_margin
    rx_pilot_search_area = rx_raw[pilot_chunk_start: pilot_chunk_end]

    #Prepare Reference
    tx_pilot_ref = np.array(ref_data['pilot_ref_real']).astype(complex) + 1j * np.array(ref_data['pilot_ref_imag']).astype(complex)
    tx_pilot_no_cp = waveform.remove_cp(tx_pilot_ref, cp_len=ofdm_conf.CP_LEN)

    #Estimate
    best_cfo, best_delay_rel, heatmap = cfo.estimate_cfo(
        tx_ref = tx_pilot_ref,
        rx_signal = rx_pilot_search_area,
        fs = 100e6,
        n_bins = 2 ** 14
    )
    best_cfo = best_cfo * -1

    #Global Correction
    actual_pilot_start = pilot_chunk_start + best_delay_rel
    refined_packet_start = actual_pilot_start - sync_len

    #Apply CFO to enite signal
    time_vec = np.arange(len(rx_raw)) / ofdm_conf.FS
    correction_vector = np.exp(-1j * 2 * np.pi * best_cfo * time_vec)
    rx_corrected = rx_raw * correction_vector

    #---------- Extract Symbols ----------
    sym_len = ofdm_conf.N + ofdm_conf.CP_LEN
    total_symbols = 1 + 1 + n_payload_syms # (1 and 1 for Sync and Pilot) This needs to up updated eventually for dynamic 
    total_samlpes = sym_len * total_symbols
    if refined_packet_start + total_samlpes > len(rx_raw):
        print(f"[Error] Packet end index {refined_packet_start + total_samlpes} exceeds file size{len(rx_raw)}")
        print(f"Setting start idx to 0")
        refined_packet_start = 0

    packet_time = rx_corrected[refined_packet_start: refined_packet_start + total_samlpes]
    all_symbols = np.split(packet_time, total_symbols)
    rx_sync_sym = all_symbols[0]
    rx_pilot_sym = all_symbols[1]
    rx_payload_syms = all_symbols[2:]

    #-------- Pilot CFO Calc --------------
    tx_pilot_ref = np.array(ref_data['pilot_ref_real']).astype(complex) + 1j * np.array(ref_data['pilot_ref_imag']).astype(complex)
    tx_pilot_no_cp = waveform.remove_cp(tx_pilot_ref, cp_len=ofdm_conf.CP_LEN)


    # ------ Channel Estimation Calc -----------t
    rx_pilot_sym_no_cp = waveform.remove_cp(rx_pilot_sym, cp_len=ofdm_conf.CP_LEN)
    rx_pilot_freq = waveform.time_to_freq(rx_pilot_sym_no_cp)
    tx_pilot_freq = waveform.time_to_freq(tx_pilot_no_cp)
    Lambda_est = CHEST.channel_estimation_calc(rx_pilot_freq=rx_pilot_freq, tx_pilot_ref=tx_pilot_freq, config=ofdm_conf)
    
    #----- Payload Extraction ---------
    pilots_idx = ofdm_conf._idx(np.array(ofdm_conf.pilot_carriers))
    tx_pilot_vals = preamble.generate_pilot_vals(config=ofdm_conf)
    demodulated_data = []

    for sym_time in rx_payload_syms:
        sym_no_cp = waveform.remove_cp(sym_time, cp_len=ofdm_conf.CP_LEN)
        sym_freq = waveform.time_to_freq(sym_no_cp)
        sym_eq = CHEST.apply_gains(sym_freq, Lambda_est=Lambda_est)

        # cal and apply phase drift
        rx_pilots_eq = sym_eq[pilots_idx]
        correlation = np.vdot(tx_pilot_vals, rx_pilots_eq)
        phase_drift = np.angle(correlation)
        phase_correction = np.exp(-1j * phase_drift)
        sym_final = sym_eq * phase_correction

        data_only = payload.extract_data(sym_final, config=ofdm_conf)
        demodulated_data.extend(data_only)
        
    demodulated_data = np.array(demodulated_data)
    demodulated_data = demodulated_data*np.sqrt(10)
    return demodulated_data, ref_data, refined_packet_start

#TODO: NEEDS TESTS
def extract_packet(rx_data:np.ndarray, start_idx:int, total_symbols:int, ofdm_conf:OFDMConfig)->np.ndarray:
    """
    Removes leading and trialing zeros from the signal

        Args:
            rx_data: Recieved RX IQ data from .dat file
            start_idx: Packet start index, get from unpack_rx_file()
            total_symbols: Total number of sync, pilot, and data symbols
            ofdm_conf: OFDMConfig data class
            
        Returns:
            Sclies raw rx_data and returns only the actual packet iq data.
    """
    sym_len = ofdm_conf.N + ofdm_conf.CP_LEN
    total_samples = sym_len * total_symbols
    start = int(start_idx)
    end = int(start_idx + total_samples)
    if (start < 0): start = 0
    if (end > len(rx_data)): end = len(rx_data)
    return rx_data[start:end]

def normalize_rx_signal(rx_data:np.ndarray)->np.ndarray:
    """
    Normalizes raw rx IQ data between +-0.9

    Args:
        rx_data: raw recieved IQ data from .dat file.
    
    Returns:
        rx_data normalized between +- 0.9

    """
    max_val = np.max(np.abs(rx_data))
    if max_val == 0:
        raise ValueError("Cannot normalize signal: max amplitude is zero.  Check the .dat file for bad capture.")
    return rx_data * (0.9) / max_val
  
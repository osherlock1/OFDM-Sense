import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
#Internal
from ofdm.config import OFDMConfig
from ofdm.core import sync, waveform, preamble, payload
from ofdm.viz import plotter
from ofdm.channel import CHEST, cfo

def main():
    parser = argparse.ArgumentParser(description="Unpack and Plot Recieved OFDM Packet")
    parser.add_argument('--file', type=str, default="rand_ofdm_packet_rx.dat", help="File name of packet to unpack")
    parser.add_argument('--ref', type=str, default="rand_ofdm_packet_ref.json", help="Reference packet json file name")
    parser.add_argument('--sim', type=bool, default=False, help="Choose to simulation (True = Use TX File)")
    args = parser.parse_args()

    ofdm_conf = OFDMConfig()

    # 1. Load Data
    print(f"Loading RX data from data_files/{args.file}...")
    if args.sim == False:
        rx_raw = np.fromfile(f"data_files/{args.file}", dtype=np.complex64)
    else: 
        rx_raw = np.fromfile(f"data_files/rand_ofdm_packet.dat", dtype=np.complex64)

    with open(f"data_files/{args.ref}") as f:
        ref_data = json.load(f)
    
    sync_ref_real = np.array(ref_data['sync_ref_real']).astype(complex)
    sync_ref_imag = np.array(ref_data["sync_ref_imag"]).astype(complex)
    sync_ref_time = sync_ref_real + 1j * sync_ref_imag
    n_payload_syms = ref_data["n_data_symb"]

    # (Optional) Add artificial CFO for testing
    # CFO = 2000 
    # t = np.arange(len(rx_raw)) / ofdm_conf.FS
    # rx_raw = rx_raw * np.exp(1j * 2 * np.pi * CFO * t)

    # ---------------------------------------------
    # 2. Synchronization (Coarse)
    # ---------------------------------------------
    M, P = sync.calculate_schmidl_cox_metrics(rx_signal=rx_raw, config=ofdm_conf)
    
    start_idx = sync.find_start_idx(
        M_metric=M, config=ofdm_conf, rx_signal=rx_raw, known_sync_time=sync_ref_time
    )

    # ---------------------------------------------
    # 3. Fine CFO Estimation
    # ---------------------------------------------
    # Corrected lengths
    sync_len = ofdm_conf.CP_LEN + ofdm_conf.N
    pilot_len = ofdm_conf.N + ofdm_conf.CP_LEN  # <--- FIXED BUG HERE

    coarse_pilot_start = start_idx + sync_len

    # Create Search Window
    search_margin = 10
    pilot_chunk_start = coarse_pilot_start - search_margin
    pilot_chunk_end = coarse_pilot_start + pilot_len + search_margin
    
    # Safety slice check
    if pilot_chunk_end > len(rx_raw):
        print("Error: File too short")
        return

    rx_pilot_search_area = rx_raw[pilot_chunk_start:pilot_chunk_end]

    # Prepare Reference
    tx_pilot_ref = np.array(ref_data['pilot_ref_real']) + 1j * np.array(ref_data['pilot_ref_imag'])
    tx_pilot_no_cp = waveform.remove_cp(tx_pilot_ref, cp_len=ofdm_conf.CP_LEN)

    # Estimate
    best_cfo, best_delay_rel, heatmap = cfo.estimate_cfo(
        tx_ref=tx_pilot_no_cp,
        rx_signal=rx_pilot_search_area,
        fs=ofdm_conf.FS,
        n_bins=4096
    )
    print(f"Estimated CFO: {best_cfo:.2f} Hz")

    # ---------------------------------------------
    # 4. Global Correction
    # ---------------------------------------------
    actual_pilot_start = pilot_chunk_start + best_delay_rel
    refined_packet_start = actual_pilot_start - sync_len

    # Apply CFO to ENTIRE signal
    time_vec = np.arange(len(rx_raw)) / ofdm_conf.FS
    correction_vector = np.exp(-1j * 2 * np.pi * best_cfo * time_vec)
    rx_corrected = rx_raw * correction_vector

    # ---------------------------------------------
    # 5. Extraction & Splitting
    # ---------------------------------------------
    sym_len = ofdm_conf.N + ofdm_conf.CP_LEN
    total_symbols = 1 + 1 + n_payload_syms
    total_samples = sym_len * total_symbols # Fixed typo 'total_samlpes'

    if refined_packet_start + total_samples > len(rx_corrected):
        print(f"[Error] Packet end exceeds file size")
        return

    # Slice Corrected Data
    packet_time = rx_corrected[refined_packet_start : refined_packet_start + total_samples]
    all_symbols = np.split(packet_time, total_symbols)

    rx_pilot_sym = all_symbols[1]
    rx_payload_syms = all_symbols[2:]

    print(f"[Success] Packet Extracted. Payload Symbols: {len(rx_payload_syms)}")

    # ---------------------------------------------
    # 6. Channel Estimation
    # ---------------------------------------------
    rx_pilot_sym_no_cp = waveform.remove_cp(rx_pilot_sym, cp_len=ofdm_conf.CP_LEN)
    rx_pilot_freq = waveform.time_to_freq(rx_pilot_sym_no_cp)
    
    tx_pilot_freq = waveform.time_to_freq(tx_pilot_no_cp)

    Lambda_est = CHEST.channel_estimation_calc(rx_pilot_freq, tx_pilot_freq, config=ofdm_conf)

    # ---------------------------------------------
    # 7. Payload Demodulation & Plotting
    # ---------------------------------------------
    demodulated_data = []
    
    for sym_time in rx_payload_syms:
        # FFT
        sym_no_cp = waveform.remove_cp(sym_time, cp_len=ofdm_conf.CP_LEN)
        sym_freq = waveform.time_to_freq(sym_no_cp)
        
        # Equalize
        sym_chest = CHEST.apply_gains(sym_freq, Lambda_est=Lambda_est)
        
        # Extract Data
        data_only = payload.extract_data(sym_chest, config=ofdm_conf)
        demodulated_data.extend(data_only)

    # Plot Results
    plotter.plot_constellation(np.array(demodulated_data), title="Final Equalized Constellation")
    plt.show()

if __name__ == "__main__":
    main()

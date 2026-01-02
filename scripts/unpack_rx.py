import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
#Internal
from ofdm.config import OFDMConfig
from ofdm.core import sync, waveform
from ofdm.viz import plotter
from ofdm.channel import CHEST


def main():
    parser = argparse.ArgumentParser(description="Unpack and Plot Recieved OFDM Packet")
    parser.add_argument('--file', type=str, default="rand_ofdm_packet_rx.dat", help="File name of packet to unpack")
    parser.add_argument('--ref', type=str, default ="rand_ofdm_packet_ref.json", help ="Reference packet json file name")
    args = parser.parse_args()

    #Load Configuration
    ofdm_conf = OFDMConfig()

    #Load Data
    print(f"Loading RX data from data_files/{args.file}...")
    rx_raw = np.fromfile(f"data_files/{args.file}", dtype=np.complex64)

    print(f"Loading Referense Data from data_files/{args.ref}...")
    with open(f"data_files/{args.ref}") as f:
        ref_data = json.load(f)
    
    #Unpack Referense Sync Symbol
    sync_ref_real = np.array(ref_data['sync_ref_real']).astype(complex)
    sync_ref_imag = np.array(ref_data["sync_ref_imag"]).astype(complex)
    sync_ref_time = sync_ref_real + 1j * sync_ref_imag

    n_payload_syms = ref_data["n_data_symb"]

    # ---------- Syncronization -------------
    #Calc M and P Metrics
    M, P = sync.calculate_schmidl_cox_metrics(rx_signal=rx_raw, config=ofdm_conf)

    #Find Start Idx
    start_idx = sync.find_start_idx(
        M_metric=M,
        config=ofdm_conf,
        rx_signal=rx_raw,
        known_sync_time=sync_ref_time
    )

    #Estimate Coarse CFO
    symbol_start_idx = start_idx + ofdm_conf.CP_LEN
    max_P = P[symbol_start_idx]
    coarse_cfo = sync.estimate_cfo_coarse(max_P, config=ofdm_conf)

    # Apply Coarse CFO Correction
    print("Applying coarse CFO correction")
    t = np.arange(len(rx_raw)) / ofdm_conf.FS
    cfo_correction = np.exp(-1j * 2*np.pi * coarse_cfo * t)

    rx_corrected = rx_raw * cfo_correction



    print(f"[Test] Coarse CFO:{coarse_cfo}, Start Idx:{start_idx}")

    #---------- Extract Symbols ----------
    sym_len = ofdm_conf.N + ofdm_conf.CP_LEN
    total_symbols = 1 + 1 + n_payload_syms # (1 and 1 for Sync and Pilot) This needs to up updated eventually for dynamic 
    total_samlpes = sym_len * total_symbols

    #Safety Check
    if start_idx + total_samlpes > len(rx_corrected):
        print(f"[Error] Packet end index {start_idx + total_samlpes} exceeds file size{len(rx_corrected)}")
        return

    #Sclice the packet
    packet_time = rx_corrected[start_idx : start_idx + total_samlpes]

    #Split into symbols
    all_symbols = np.split(packet_time, total_symbols)

    rx_sync_sym = all_symbols[0]
    rx_pilot_sym = all_symbols[1]
    rx_payload_syms = all_symbols[2:]

    print(f"[Success] Packet Extraacted.")
    print(f"  -> {len(rx_payload_syms)} Payload Symbols extracted")

    plotter.plot_time_series(np.abs(rx_pilot_sym), title="Pilot Symbol")
    plt.show()

    # ------ Channel Estimation -----------
    #Remove CP
    rx_pilot_sym_no_cp = waveform.remove_cp(rx_pilot_sym, cp_len = ofdm_conf.CP_LEN)
    rx_pilot_freq = waveform.time_to_freq(rx_pilot_sym_no_cp)

    #Get TX pilot Ref
    tx_pilot_ref = np.array(ref_data['pilot_ref_real']).astype(complex) + 1j * np.array(ref_data['pilot_ref_imag']).astype(complex)
    tx_pilot_no_cp = waveform.remove_cp(tx_pilot_ref, cp_len=ofdm_conf)
    tx_pilot_freq = waveform.time_to_freq(tx_pilot_no_cp)
    #Calculate Channel Gains
    Lambda_est = CHEST.channel_estimation_calc(rx_pilot_freq=rx_pilot_freq, tx_pilot_ref=tx_pilot_freq, config=ofdm_conf)

    #Plot Channel Gains
    plt.figure()
    plt.plot(np.abs(Lambda_est))
    plt.title("Lambda ABS")

    plt.figure()
    plt.plot(np.angle(Lambda_est))
    plt.title("Lambda Angle")
    plt.show()
    
if __name__ == "__main__":
    main()
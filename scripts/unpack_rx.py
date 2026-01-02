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
    parser.add_argument('--ref', type=str, default ="rand_ofdm_packet_ref.json", help ="Reference packet json file name")
    parser.add_argument('--sim', type=bool, default = False, help="Choose to simulation (True = Use TX File)")
    args = parser.parse_args()

    #Load Configuration
    ofdm_conf = OFDMConfig()

    #Load Data
    print(f"Loading RX data from data_files/{args.file}...")
    if args.sim == False:
        rx_raw = np.fromfile(f"data_files/{args.file}", dtype=np.complex64)
    else: 
        rx_raw = np.fromfile(f"data_files/rand_ofdm_packet.dat", dtype=np.complex64)

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


    #---------- Extract Symbols ----------
    sym_len = ofdm_conf.N + ofdm_conf.CP_LEN
    total_symbols = 1 + 1 + n_payload_syms # (1 and 1 for Sync and Pilot) This needs to up updated eventually for dynamic 
    total_samlpes = sym_len * total_symbols

    #Safety Check
    if start_idx + total_samlpes > len(rx_raw):
        print(f"[Error] Packet end index {start_idx + total_samlpes} exceeds file size{len(rx_raw)}")
        return

    #Sclice the packet
    packet_time = rx_raw[start_idx : start_idx + total_samlpes]

    #Split into symbols
    all_symbols = np.split(packet_time, total_symbols)

    rx_sync_sym = all_symbols[0]
    rx_pilot_sym = all_symbols[1]
    rx_payload_syms = all_symbols[2:]

    print(f"[Success] Packet Extraacted.")
    print(f"  -> {len(rx_payload_syms)} Payload Symbols extracted")

    #-------- Pilot CFO Calc --------------
    tx_pilot_ref = np.array(ref_data['pilot_ref_real']).astype(complex) + 1j * np.array(ref_data['pilot_ref_imag']).astype(complex)
    tx_pilot_no_cp = waveform.remove_cp(tx_pilot_ref, cp_len=ofdm_conf.CP_LEN)

    #Calculate CFO
    best_cfo, best_delay, heatmap = cfo.estimate_cfo(
        tx_ref = tx_pilot_no_cp,
        rx_signal=rx_pilot_sym,
        fs=ofdm_conf.FS,
        search_window=10,
        n_bins=4096
    )
    print(f"Calculated CFO:{best_cfo}, Calculated Delay:{best_delay}")

    #Apply CFO
    corrected_pilot = cfo.apply_cfo(rx_signal=rx_pilot_sym[best_delay: best_delay + ofdm_conf.N], cfo=best_cfo, fs=ofdm_conf.FS)


    # ------ Channel Estimation Calc -----------
    #Remove CP of ref pilot
    rx_pilot_freq = waveform.time_to_freq(corrected_pilot)

    #Get TX pilot Ref
    tx_pilot_freq = waveform.time_to_freq(tx_pilot_no_cp)
    #Calculate Channel Gains
    Lambda_est = CHEST.channel_estimation_calc(rx_pilot_freq=rx_pilot_freq, tx_pilot_ref=tx_pilot_freq, config=ofdm_conf)

    #Plot Channel Gains
    plt.figure()
    plt.plot(np.fft.fftshift(np.abs(Lambda_est)[ofdm_conf.data_carriers]))
    plt.title("Lambda ABS")

    plt.figure()
    plt.plot(np.angle(Lambda_est[ofdm_conf.data_carriers]))
    plt.title("Lambda Angle")
    plt.show()
    
    #----- Payload Extraction ---------
    demodulated_data = []
    for sym_time in rx_payload_syms:
        #FFT
        sym_no_cp = waveform.remove_cp(sym_time, cp_len=ofdm_conf.CP_LEN)
        sym_freq = waveform.time_to_freq(sym_no_cp)

        #Apply Channel gain
        sym_chest = CHEST.apply_gains(sym_freq, Lambda_est=Lambda_est)

        #Extract Data Bins
        data_only = payload.extract_data(sym_chest, config=ofdm_conf)
        demodulated_data.extend(data_only)
    
    plt.figure()
    plt.scatter(np.real(demodulated_data), np.imag(demodulated_data), alpha=0.5)
    plt.title="Constalation plot"
    plt.show()





if __name__ == "__main__":
    main()
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
#Internal
from ofdm.config import OFDMConfig
from ofdm.core import sync, waveform, preamble, payload
from ofdm.viz import plotter
from ofdm.channel import CHEST, cfo
from ofdm.utils import eval
from ofdm.modulation import qam



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

    #Find pilot symbol
    sync_len = ofdm_conf.CP_LEN + ofdm_conf.N
    pilot_len = ofdm_conf.CP_LEN + ofdm_conf.N

    coarse_pilot_start = start_idx + sync_len

    #Create a search window
    search_margin = 10
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
        fs = ofdm_conf.FS,
        n_bins = 4096
    )
    #FIXME: Using Opposite Sign on CFO since it seems to be revered(need to look into)
    best_cfo = best_cfo * -1
    print(f"Estimated CFO:{best_cfo}, Best Delay:{best_delay_rel}")


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

    #Safety Check
    if start_idx + total_samlpes > len(rx_raw):
        print(f"[Error] Packet end index {start_idx + total_samlpes} exceeds file size{len(rx_raw)}")
        return

    #Sclice the packet
    packet_time = rx_corrected[refined_packet_start: refined_packet_start + total_samlpes]

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

    print(f"Calculated CFO:{best_cfo}, Calculated Delay:{best_delay_rel}")



    # ------ Channel Estimation Calc -----------
    #Remove CP of ref pilot
    rx_pilot_sym_no_cp = waveform.remove_cp(rx_pilot_sym)
    rx_pilot_freq = waveform.time_to_freq(rx_pilot_sym_no_cp)

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
    pilots_idx = ofdm_conf._idx(np.array(ofdm_conf.pilot_carriers))
    tx_pilot_vals = preamble.generate_pilot_vals(config=ofdm_conf)
    demodulated_data = []

    for sym_time in rx_payload_syms:
        #Remove CP
        sym_no_cp = waveform.remove_cp(sym_time, cp_len=ofdm_conf.CP_LEN)
        
        #Convert to Freq Domain
        sym_freq = waveform.time_to_freq(sym_no_cp)

        #Apply Channel Gains
        sym_eq = CHEST.apply_gains(sym_freq, Lambda_est=Lambda_est)

        #Estimate Phase Drift
        rx_pilots_eq = sym_eq[pilots_idx]
        correlation = np.vdot(tx_pilot_vals, rx_pilots_eq)
        phase_drift = np.angle(correlation)

        #Apply Phase correction
        phase_correction = np.exp(-1j * phase_drift)
        sym_final = sym_eq * phase_correction

        print(f"Symbol Drift {np.degrees(phase_drift):.2f} degrees")

        #Extract Data
        data_only = payload.extract_data(sym_final, config=ofdm_conf)
        demodulated_data.extend(data_only)
        


    #Final Data
    demodulated_data = np.array(demodulated_data)
    demodulated_data = demodulated_data*np.sqrt(10)
    

    #--------- Evaluation ---------
    #Get Referense Data
    ref_binary = ref_data['binary_data']
    n_ref_samples = ref_data['n_samples']
    ref_iq = binary_ref_to_iq(binary_string=ref_binary, n_samples=n_ref_samples)

    #Calculate EVM
    evm = eval.calc_EVM(iq_rx=demodulated_data, iq_ref=ref_iq)
    print(f"EVM:{evm}")

    plt.figure()
    plt.scatter(np.real(demodulated_data), np.imag(demodulated_data), alpha=0.5)
    plt.title="Constalation plot"
    plt.show()



def binary_ref_to_iq(binary_string:str, n_samples:int)->np.ndarray:

    full_string = "".join(binary_string)
    print(f"Original Binary String {full_string}")

    #Parse String into 4 bit words
    word_len = 4
    binary_word_list = np.array([full_string[i:word_len + i] for i in range(0 ,len(full_string), word_len)])
    
    #Convert to IQ
    iq_array = [qam.binary_to_iq(word) for word in binary_word_list]
    return np.array(iq_array) * np.sqrt(10)
    
    

    



if __name__ == "__main__":
    main()
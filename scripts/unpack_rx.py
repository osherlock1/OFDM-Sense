import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import os

#Internal
from ofdm.config import OFDMConfig
from ofdm.core import sync, waveform, preamble, payload
from ofdm.viz import plotter
from ofdm.channel import CHEST, cfo
from ofdm.utils import eval
from ofdm.modulation import qam
from ofdm.utils import usrp


def unpack_rx_file(ofdm_conf:OFDMConfig, rx_path:str, ref_path:str, sim:bool = False)->np.ndarray:
    """  
    Take raw ofdm binary data from the rx and unpack it including syncronization, channel estimation, and cfo calibration.
    """

    #Load Data
    print(f"Loading RX data from {rx_path}...")
    if sim == False:
        rx_raw = np.fromfile(rx_path, dtype=np.complex64)
    else: 
        rx_raw = np.fromfile(rx_path, dtype=np.complex64)
    print(f"Loading Referense Data from {ref_path}...\n")
    with open(ref_path) as f:
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
        known_sync_time=sync_ref_time,
        search_window=500
    )

    #Find pilot symbol
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
    #FIXME: Using Opposite Sign on CFO since it seems to be revered(need to look into)
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

    #Safety Check
    if refined_packet_start + total_samlpes > len(rx_raw):
        print(f"[Error] Packet end index {refined_packet_start + total_samlpes} exceeds file size{len(rx_raw)}")
        print(f"Setting start idx to 0")
        refined_packet_start = 0
        #return

    #Sclice the packet
    packet_time = rx_corrected[refined_packet_start: refined_packet_start + total_samlpes]

    #Split into symbols
    all_symbols = np.split(packet_time, total_symbols)

    rx_sync_sym = all_symbols[0]
    rx_pilot_sym = all_symbols[1]
    rx_payload_syms = all_symbols[2:]

    #-------- Pilot CFO Calc --------------
    tx_pilot_ref = np.array(ref_data['pilot_ref_real']).astype(complex) + 1j * np.array(ref_data['pilot_ref_imag']).astype(complex)
    tx_pilot_no_cp = waveform.remove_cp(tx_pilot_ref, cp_len=ofdm_conf.CP_LEN)


    # ------ Channel Estimation Calc -----------
    #Remove CP of ref pilot
    rx_pilot_sym_no_cp = waveform.remove_cp(rx_pilot_sym, cp_len=ofdm_conf.CP_LEN)
    rx_pilot_freq = waveform.time_to_freq(rx_pilot_sym_no_cp)

    #Get TX pilot Ref
    tx_pilot_freq = waveform.time_to_freq(tx_pilot_no_cp)
    #Calculate Channel Gains
    Lambda_est = CHEST.channel_estimation_calc(rx_pilot_freq=rx_pilot_freq, tx_pilot_ref=tx_pilot_freq, config=ofdm_conf)
    
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

        #Extract Data
        data_only = payload.extract_data(sym_final, config=ofdm_conf)
        demodulated_data.extend(data_only)
        
    #Final Data
    demodulated_data = np.array(demodulated_data)
    demodulated_data = demodulated_data*np.sqrt(10)

    print(f"[Success] Packet Extracted.")
    print(f"  -> {len(rx_payload_syms)} Payload Symbols extracted")
    print(f"Calculated CFO:{best_cfo}.\n")

    return demodulated_data, ref_data, refined_packet_start

def main():
    parser = argparse.ArgumentParser(description="Unpack and Plot Recieved OFDM Packet")
    parser.add_argument('--file', type=str, default="./data_files/rand_ofdm_packet_rx.dat", help="File name of packet to unpack")
    parser.add_argument('--ref', type=str, default ="./data_files/rand_ofdm_packet_ref.json", help ="Reference packet json file name")
    parser.add_argument('--sim', type=bool, default = False, help="Choose to simulation (True = Use TX File)")
    parser.add_argument('--plot', action="store_true", help="Plot Constalation Diagrams of Unpacked OFDM Packets")
    args = parser.parse_args()

    #Load Configurations
    USRP_CONFIG_PATH = "./configs/usrp_settings.yaml"
    usrp_conf = usrp.load_config(USRP_CONFIG_PATH)
    ofdm_conf = OFDMConfig()
    
    rx_channel_idx = usrp_conf.rx_channel_idx.replace(",", "")
    
    print(f"Unpacking {len(rx_channel_idx)} RX Channels...")

    #Unpack all RX files
    demodulated_dict = {}
    start_idx_list = []
    for channel in rx_channel_idx: 
        if len(rx_channel_idx) != 1:
            rx_path = f"./data_files/rand_ofdm_packet_rx.0" + channel + ".dat"
        print(f"#####################")        
        print(f"Unpacking Channel" + channel + "...")
        print(f"#####################\n")  
        demodulated_data, ref_data, start_idx = unpack_rx_file(ofdm_conf=ofdm_conf, rx_path=rx_path, ref_path=args.ref)
        demodulated_dict[f"Channel_" + channel] = demodulated_data
        start_idx_list.append((channel, start_idx))

    #--------- Evaluation ---------
    print(f"#####################") 
    print(f"Calculating OFDM Perforamnce")
    print(f"#####################") 
    #Get Referense Data
    ref_binary = ref_data['binary_data']
    n_ref_samples = ref_data['n_samples']
    ref_iq = binary_ref_to_iq(binary_string=ref_binary, n_samples=n_ref_samples)

    #Store Metrics for post processing
    ber_list = [] 
    ser_list = []
    evm_list = []

    #Eval and plot all unpacked data
    for channel_name, demodulated_data in demodulated_dict.items():
        print(f"Performance metrics for {channel_name}")

        #Calculate EVM
        evm = eval.calc_EVM(iq_rx=demodulated_data, iq_ref=ref_iq)
        print(f"EVM:{evm:.2f}dB")

        #Calculate SER
        ser = eval.calc_SER(iq_rx=demodulated_data, iq_ref=ref_iq)
        print(f"SER:{ser * 100:.2f}%")

        #Calculate BER
        ber = eval.calc_BER(iq_rx = demodulated_data, iq_ref=ref_iq)
        print(f"BER:{ber*100:.2f}% \n")
        
        ber_list.append(ber)
        ser_list.append(ser)
        evm_list.append(evm)

    for channel_name, demodulated_data in demodulated_dict.items():
        #----------- Save Unpacked Data ---------- 
        unpacked_file_name = f"unpacked_data_{channel_name}.json"
        save_unpacked_data(demodulated_data, file_name=unpacked_file_name)

    if (args.plot):
        # ---------- Plots ---------------
        for channel_name, demodulated_data in demodulated_dict.items():
            ref_constalation = qam.get_reference_constalation()

            plt.figure()
            plt.scatter(np.real(demodulated_data), np.imag(demodulated_data), alpha=0.5)
            plt.scatter(np.real(ref_constalation), np.imag(ref_constalation))
            plt.title(f"{channel_name} Constalation plot")
            plt.xlabel("Real")
            plt.ylabel("Imaginary")
        plt.show()


    eval_metrics_path = "./data_files/ofdm_performance.json"
    os.makedirs(os.path.dirname(eval_metrics_path), exist_ok=True)
    json_data = {
        "evm":evm_list,
        "ser":ser_list,
        "ber":ber_list,
        "start_idx":start_idx_list
    }
    with open(eval_metrics_path, "w") as f:
        json.dump(json_data, f, indent=2, default=float)
    print(f"Stored Eval Metrics to {eval_metrics_path}")


    


def binary_ref_to_iq(binary_string:str, n_samples:int)->np.ndarray:

    full_string = "".join(binary_string)

    #Parse String into 4 bit words
    word_len = 4
    binary_word_list = np.array([full_string[i:word_len + i] for i in range(0 ,len(full_string), word_len)])
    
    #Convert to IQ
    iq_array = [qam.binary_to_iq(word) for word in binary_word_list]
    return np.array(iq_array) * np.sqrt(10)
    
    
def save_unpacked_data(rx_iq:np.ndarray, file_name:str):
    """  
    Saves the unpacked data a json file

    Args:
        rx_iq: Unpacked RX iq data (scaled +-3)
        file_name: name of file to save data (do not include data_files/)
    """
    #Convert IQ to Binary
    binary_rx = [qam.iq_to_binary(sample) for sample in rx_iq]
    
    #Convert rx_iq to list
    

    unpacked_data = {
        "unpacked_data_real":np.real(rx_iq).tolist(),
        "unpacked_data_imag":np.imag(rx_iq).tolist(),
        "unpacked_binary_data":binary_rx
    }

    json_path = f"data_files/{file_name}"
    with open(json_path, "w") as f:
        json.dump(unpacked_data, f, indent=2)
    print(f"[Success] Saved Unpacked Data to {json_path}")

    
    

    
    



if __name__ == "__main__":
    main()

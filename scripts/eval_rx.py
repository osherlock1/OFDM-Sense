import numpy as np
import json
import matplotlib.pyplot as plt
#internal
from ofdm.config import OFDMConfig
from ofdm.modulation import qam
from ofdm.utils import eval

def main():
    ofdm_conf = OFDMConfig()

    #Get RX Data
    rx_data_file_name = "unpacked_data.json"
    rx_data = unpack_json(rx_data_file_name)
    
    #Get Ref Data
    ref_data_file_name = "rand_ofdm_packet_ref.json"
    ref_data = unpack_json(ref_data_file_name)

    #Unpack Json Files
    rx_iq = np.array(rx_data["unpacked_data_real"]) + 1j * np.array(rx_data["unpacked_data_imag"])
    rx_binary = rx_data["unpacked_binary_data"]
    
    #Unpack Ref Data
    ref_binary_string = ref_data['binary_data']
    n_samples = ref_data['n_samples']
    n_sym = ref_data['n_data_symb']
    ref_iq = binary_ref_to_iq(binary_string=ref_binary_string, n_samples=n_samples)
    
    #sc_metrics = per_subcarrier_eval(rx_iq_data = rx_iq, ref_iq_data = ref_iq, config=ofdm_conf, n_sym = n_sym)
    diagnose_spectrum(rx_iq_data=rx_iq, ref_iq_data=ref_iq, config=ofdm_conf)

def per_subcarrier_eval(rx_iq_data:np.ndarray, ref_iq_data:np.ndarray, config:OFDMConfig, n_sym:int):
    """  
    Evaluate the Performance of Individual Subcarriers
    """

    n_data_carriers = len(config.data_carriers)

    rx_grid = rx_iq_data.reshape(-1, n_data_carriers)
    ref_grid = ref_iq_data.reshape(-1, n_data_carriers)
    
    sc_metrics = {
        "indices":[],
        "evm_db": []
    }

    print(f"Evaluating {n_data_carriers} subcarriers...")

    for col_idx, sc_index in enumerate(config.data_carriers):
        #Extract the COlUMN
        rx_col = rx_grid[:, col_idx]
        ref_col = ref_grid[:, col_idx]

        #Calculate Metric for this subcarrier
        evm = eval.calc_EVM(iq_rx=rx_col, iq_ref=ref_col)

        #Store Results
        sc_metrics['indices'].append(sc_index)
        sc_metrics['evm_db'].append(evm)
    
    
    k = sc_metrics['indices']
    evm = sc_metrics['evm_db']

    db_threshold = -4

    for i, metric in enumerate(evm):
        if metric >= db_threshold:
            print(f"Subcarrier{k[i]}, EVM:{metric:.2f}dB")
    plt.figure()
    plt.stem(k, evm)
    plt.title("EVM per subcarrier")
    plt.show()

    


    
    

    

def unpack_json(json_file_name:str)->dict:
    """
    Unpacks a json file and returns the json dictionary
    
    Args:
        json_file_name: name of file in the data_files dir (do not include data_files/)
    
    Returns:
        json data dictionary
    """
    json_file_path = f"data_files/{json_file_name}"
    with open(json_file_path, "r") as f:
        data = json.load(f)
    print(f"[Success] Unpacked {json_file_path}")
    return data

def binary_ref_to_iq(binary_string:str, n_samples:int)->np.ndarray:

    full_string = "".join(binary_string)

    #Parse String into 4 bit words
    word_len = 4
    binary_word_list = np.array([full_string[i:word_len + i] for i in range(0 ,len(full_string), word_len)])
    
    #Convert to IQ
    iq_array = [qam.binary_to_iq(word) for word in binary_word_list]
    return np.array(iq_array) * np.sqrt(10)



def diagnose_spectrum(rx_iq_data: np.ndarray, ref_iq_data: np.ndarray, config: OFDMConfig):
    """
    Plots the average magnitude of every subcarrier to spot interference/fading.
    """
    n_carriers = len(config.data_carriers)
    
    # Reshape to (Symbols, Subcarriers)
    rx_grid = rx_iq_data.reshape(-1, n_carriers)
    ref_grid = ref_iq_data.reshape(-1, n_carriers)

    # Calculate Average Magnitude (Power) per subcarrier
    # We use median to ignore random noise spikes
    rx_mag = np.median(np.abs(rx_grid), axis=0)
    ref_mag = np.median(np.abs(ref_grid), axis=0)
    
    # Calculate the Difference (Gain/Loss)
    gain_diff = 20 * np.log10(rx_mag / (ref_mag + 1e-12))
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # 1. Top: Magnitude Comparison
    plt.subplot(2, 1, 1)
    plt.title("Spectrum Magnitude Check")
    plt.plot(config.data_carriers, ref_mag, label="Reference (TX)", marker='o', alpha=0.5)
    plt.plot(config.data_carriers, rx_mag, label="Received (RX)", marker='x')
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    
    # 2. Bottom: Gain Error (Should be flat 0 dB)
    plt.subplot(2, 1, 2)
    plt.title("Gain Error per Subcarrier")
    plt.stem(config.data_carriers, gain_diff, basefmt=" ")
    plt.axhline(0, color='k')
    plt.ylabel("Gain Error (dB)")
    plt.xlabel("Subcarrier Index")
    plt.grid(True)
    
    # Highlight the "bad" ones you listed
    bad_indices = [4, 25, 27, 50, 84, 126]
    # Ensure carriers is a numpy array for proper indexing
    carriers_arr = np.array(config.data_carriers)
    
    # Create a mask: True where the carrier is in our "bad list"
    mask = np.isin(carriers_arr, bad_indices)
    
    # Extract X and Y using the mask (No loops needed!)
    bad_x = carriers_arr[mask]
    bad_y = gain_diff[mask]
    
    # Plot all suspects at once
    if len(bad_x) > 0:
        plt.plot(bad_x, bad_y, 'ro', label="Suspects")
        plt.legend()

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
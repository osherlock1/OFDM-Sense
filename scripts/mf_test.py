import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
from ofdm.channel import delay
from ofdm.modulation import qam
from ofdm.config import OFDMConfig
# =========================
# Main
# =========================
def main():

    ofdm_conf = OFDMConfig()
    
    raw_rx_data = np.fromfile("data_files/rand_ofdm_packet_rx.dat", dtype=complex)
    raw_tx_data = np.fromfile("data_files/rand_ofdm_packet.dat")


    #Scale raw_rx_data
    max_val = np.max(np.abs(raw_rx_data))

    if max_val > 0:
        scale_factor = 0.9 / max_val
        raw_rx_data = raw_rx_data * scale_factor


    #Unpack Data
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


    z, lags = delay.matched_filter_calc(rx_iq = raw_rx_data, ref_iq=raw_tx_data, fs = ofdm_conf.FS)

    delay_idx = np.argmax(z)
    caled_delay = lags[delay_idx]
    print(caled_delay)

    plt.figure()
    plt.plot(lags, np.abs(z))
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
if __name__ == "__main__":
    main()

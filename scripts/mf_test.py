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

    #Define OFDM Configuration
    ofdm_conf = OFDMConfig()
    
    #Unpack Raw RX and TX data
    raw_rx_data = np.fromfile("data_files/rand_ofdm_packet_rx.dat", dtype=np.complex64)
    raw_tx_data = np.fromfile("data_files/rand_ofdm_packet.dat", dtype=np.complex64)

    #Scale raw_rx_data
    max_val = np.max(np.abs(raw_rx_data))

    if max_val > 0:
        scale_factor = 0.9 / max_val
        raw_rx_data = raw_rx_data * scale_factor

    #Calculate Matched Filter Delay
    z, lags = delay.matched_filter_calc(rx_iq = raw_rx_data, ref_iq=raw_tx_data, fs = ofdm_conf.FS)
    z_mag = np.abs(z)
    #Find Peak of correlation
    peak_idx = np.argmax(z_mag)

    #Interpolation for fine resoluation
    window_radius = 5
    start_idx = max(0, peak_idx - window_radius)
    end_idx = min(len(z_mag), peak_idx + window_radius + 1)

    lags_window = lags[start_idx:end_idx]
    z_window = z_mag[start_idx:end_idx]

    #Create interpolator function
    f_interp = interp1d(lags_window, z_window, kind='cubic')

    #Create a high-resolution time grid
    fine_lags = np.linspace(lags_window[0], lags_window[-1], num = 10000)
    fine_z = f_interp(fine_lags)

    #Fine Exact Peak on fine grid
    fine_peak_idx = np.argmax(fine_z)
    fine_delay = fine_lags[fine_peak_idx]

    #Calculate Distance
    SPEED_OF_LIGHT = 299792458

    raw_distance = np.abs(fine_delay) * SPEED_OF_LIGHT

    print(f"Coarse Delay: {lags[peak_idx]*1e6:.4f}us")
    print(f"Fine Delay: {fine_delay*1e6:.4f}us")
    print(f"Distance: {raw_distance:.2f}meters")
    

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

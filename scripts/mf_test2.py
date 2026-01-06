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

    #Unpack TX Pilot symbol
    with open("data_files/rand_ofdm_packet_ref.json", "r") as f:
        ref_data = json.load(f)

    #Get Tx pilot symbol
    tx_pilot = np.array(ref_data['pilot_ref_real']) + 1j * np.array(ref_data['pilot_ref_imag'])

    #Scale raw_rx_data
    max_val = np.max(np.abs(raw_rx_data))

    if max_val > 0:
        scale_factor = 0.9 / max_val
        raw_rx_data = raw_rx_data * scale_factor

    #Upsample RX
    N_rx = len(raw_rx_data)
    K_rx = 100 #Scaling factor 100 Mhz -> 10 Ghz
    N_rx_padded = N_rx * K_rx

    #Convert rx to freq
    rx_freq = np.fft.fftshift(np.fft.fft(raw_rx_data))

    total_zeros = N_rx_padded - N_rx
    zeros_side = np.zeros(total_zeros // 2)

    rx_freq_padded = np.concatenate([zeros_side, rx_freq, zeros_side])
    rx_freq_ready = np.fft.ifftshift(rx_freq_padded)

    rx_upsampled = np.fft.ifft(rx_freq_ready) * K_rx

    #Upsample TX
    N_tx = len((tx_pilot))
    K_tx = 100
    N_tx_padded = N_tx * K_tx
    
    #Convert tx to freq
    tx_freq = np.fft.fftshift(np.fft.fft(tx_pilot))

    total_zeros = N_tx_padded - N_tx
    zeros_side = np.zeros(total_zeros // 2)

    tx_freq_padded = np.concatenate([zeros_side, tx_freq, zeros_side])
    tx_freq_ready = np.fft.ifftshift(tx_freq_padded)

    tx_upsampled = np.fft.ifft(tx_freq_ready) * K_tx



    #Calculate Matched Filter Delay
    z, lags = delay.matched_filter_calc(rx_iq = rx_upsampled, ref_iq=tx_upsampled, fs = (ofdm_conf.FS * 100))
    z_mag = np.abs(z)
    #Find Peak of correlation
    peak_idx = np.argmax(z_mag)

    fine_delay = lags[peak_idx]

    #Calculate Distance
    SPEED_OF_LIGHT = 299792458
    CONSTANT = -35424.5068165258
    raw_distance = (fine_delay * SPEED_OF_LIGHT) - CONSTANT

    #Calibrate
    actual_distance = 1 #1 meter

    constant = (fine_delay * SPEED_OF_LIGHT) - actual_distance
    print(f"Calculated Constant: {constant}")



    print(f"Coarse Delay: {lags[peak_idx]*1e6:.4f}us")
    print(f"Fine Delay: {fine_delay*1e6:.4f}us")
    print(f"Distance: {raw_distance:.2f}meters")


    plt.figure()
    plt.plot(lags, np.abs(z_mag))
    plt.title("Match Filter Delay Correlation Magnitude")
    plt.xlabel("Time(s)")
    plt.ylabel("Magnitude")
    


    range = 10000
    zoom_start = peak_idx - range
    zoom_end = peak_idx + range

    plt.figure()
    plt.plot(lags[zoom_start:zoom_end], np.abs(z_mag[zoom_start:zoom_end]))
    plt.title("Match Filter Delay Correlation Magnitude (Zoomed)")
    plt.xlabel("Time(s)")
    plt.ylabel("Magnitude")
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

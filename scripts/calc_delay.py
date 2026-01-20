import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
from ofdm.channel import delay
from ofdm.modulation import qam
from ofdm.config import OFDMConfig


#Config
CALIBRATION_PATH = "metadata/calibration.json"
RX_DATA_PATH = "data_files/rand_ofdm_packet.dat"
TX_REF_PATH = "data_files/rand_ofdm_packet_ref.json"

C = 299792458 #Speed of light

def main():

    #Define OFDM Configuration
    ofdm_conf = OFDMConfig()

    #Get constant
    with open(CALIBRATION_PATH, 'r') as f:
        cali_data = json.load(f)
    CONSTANT = cali_data['constant']
    
    #Unpack Raw RX and TX data
    raw_rx_data = np.fromfile("data_files/rand_ofdm_packet_rx.dat", dtype=np.complex64)

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

    #Upsampel Data
    rx_upsampled = upsample(raw_rx_data, scale_factor = 100)
    tx_upsampled = upsample(tx_pilot, scale_factor=100)

    #Calculate Matched Filter Delay
    z, lags = delay.matched_filter_calc(rx_iq = rx_upsampled, ref_iq=tx_upsampled, fs = (ofdm_conf.FS * 100))
    z_mag = np.abs(z)

    #Find Peak of correlation
    peak_idx = np.argmax(z_mag)
    fine_delay = lags[peak_idx]

    #Calculate Distance
    raw_distance = (fine_delay * C) - CONSTANT

    print(f"Coarse Delay: {lags[peak_idx]*1e6:.4f}us")
    print(f"Fine Delay: {fine_delay*1e6:.4f}us")
    print(f"Distance: {raw_distance:.2f}meters")


    plt.figure()
    plt.plot(lags, np.abs(z_mag))
    plt.title("Match Filter Delay Correlation Magnitude")
    plt.xlabel("Time(s)")
    plt.ylabel("Magnitude")
    
    range = 3000
    zoom_start = peak_idx - range
    zoom_end = peak_idx + range

    plt.figure()
    plt.plot(lags[zoom_start:zoom_end], np.abs(z_mag[zoom_start:zoom_end]))
    plt.title("Match Filter Delay Correlation Magnitude (Zoomed)")
    plt.xlabel("Time(s)")
    plt.ylabel("Magnitude")
    plt.show()

def upsample(raw_data:np.ndarray, scale_factor:int = 100)->np.ndarray:
    """
    Upsamples raw data based on the scale factor for matched filter delay esiamtion

    Args:
        raw_data: Raw RX or Tx pilot data to be upsampled
        scale_factor: Multiplication factor for upsamples i.ee to upsample from 100Mhz to 10Ghz use scale_factor = 100
    
    Returns:
        np.ndarray of upsampled data
    """
    N = len(raw_data)
    K = scale_factor
    N_padded = N * K

    #Convert to freq
    freq = np.fft.fftshift(np.fft.fft(raw_data))

    total_zeros = N_padded - N #Calculate total number of zeros for upsampling
    zeros_side = np.zeros(total_zeros // 2) #Get the number of requied zeros to append and prepend to original data

    freq_padded = np.concatenate([zeros_side, freq, zeros_side])
    freq_ready = np.fft.ifftshift(freq_padded)

    upsampled = np.fft.ifft(freq_ready) * K
    return upsampled


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

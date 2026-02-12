import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
from ofdm.channel import delay
from ofdm.modulation import qam
from ofdm.config import OFDMConfig
import scipy
import scipy.signal


#Config
CALIBRATION_PATH = "metadata/calibration.json"
RX_DATA_PATH = "data_files/rand_ofdm_packet.dat"
TX_REF_PATH = "data_files/rand_ofdm_packet_ref.json"



C = scipy.constants.c #Speed of light

def calc_delay_with_ref(raw_rx_data: np.ndarray, wired_rx_data: np.ndarray,):
    """
    Calculate delay between tx and rx using a wired references. 
    """

    pass

def calc_delay():
    """
    Docstring for calc_delay
    """
    pass



def main():
    #Define OFDM Configuration
    ofdm_conf = OFDMConfig()

    #Get constant
    with open(CALIBRATION_PATH, 'r') as f:
        cali_data = json.load(f)
    CONSTANT = cali_data['constant']
    
    #Unpack Sivers RX and TX data
    raw_rx_data = np.fromfile("data_files/rand_ofdm_packet_rx.01.dat", dtype=np.complex64)

    #Unpack wired RX data 
    wired_rx_data = np.fromfile("data_files/rand_ofdm_packet_rx.00.dat", dtype=np.complex64)

    #Unpack TX Pilot symbol
    with open("data_files/rand_ofdm_packet_ref.json", "r") as f:
        ref_data = json.load(f)

    #Get Tx pilot symbol
    tx_pilot = np.array(ref_data['pilot_ref_real']) + 1j * np.array(ref_data['pilot_ref_imag'])

    #Scale raw_rx_data
    scaled_wireless_rx = scale_rx_signal(raw_rx_data=raw_rx_data)
    scaled_wired_rx = scale_rx_signal(raw_rx_data=wired_rx_data)

    #Upsampel Data
    rx_wireless_upsampled = upsample(scaled_wireless_rx, scale_factor = 100)
    rx_wired_upsampled = upsample(scaled_wired_rx, scale_factor=100)
    tx_upsampled = upsample(tx_pilot, scale_factor=100)

    #Calculate Matched Filter Delay for wireless
    z_wireless, lags_wireless = delay.matched_filter_calc(rx_iq = rx_wireless_upsampled, ref_iq=tx_upsampled, fs = (ofdm_conf.FS * 100))
    z_mag_wireless = np.abs(z_wireless)

    #Find Peak of correlation
    peak_idx_wireless = np.argmax(z_mag_wireless)
    fine_delay_wireless = lags_wireless[peak_idx_wireless]

    #Calculate Matched Filter Delay for wired
    z_wired, lags_wired = delay.matched_filter_calc(rx_iq=rx_wired_upsampled, ref_iq=tx_upsampled,fs = (ofdm_conf.FS * 100))
    z_mag_wired = np.abs(z_wired)

    #Fine peak of correlatiokn
    peak_idx_wired = np.argmax(z_mag_wired)
    fine_delay_wired = lags_wired[peak_idx_wired]

    #print(f"Wired Delay: {fine_delay_wired}, Wireless Delay: {fine_delay_wireless}")

    #print(f"Wired Delay - Wireless Delay = {fine_delay_wired - fine_delay_wireless}")
    caled_delay = ((fine_delay_wireless - fine_delay_wired) * 1e9)
    print(f"Delay: {caled_delay:.5f}ns")

    #Calculate Distance
    #print(f"Constant used is: {CONSTANT}")
    raw_distance = ((fine_delay_wireless - fine_delay_wired) * C) - CONSTANT
    print(f"Distance: {raw_distance:.5f}")
    print(f"{caled_delay:5f},{raw_distance:.5f}")
    

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

def scale_rx_signal(raw_rx_data:np.ndarray)->np.ndarray:
    max_val = np.max(np.abs(raw_rx_data))

    if max_val > 0:
        scale_factor = 0.9 / max_val
        scaled_rx_data = raw_rx_data * scale_factor
    return scaled_rx_data


def calculate_precise_delay(rx_signal, ref_signal, fs, upsample_factor=100):
    """
    Calculates delay using coarse correlation zero-padded FFT interpolation
    """
    corr = scipy.signal.correlate(rx_signal, ref_signal, mode='full')
    lags = scipy.signal.correlation_lags(len(rx_signal), len(ref_signal), mode='full')

    # Find coarse peak
    mag = np.abs(corr)
    coarse_idx = np.argmax(mag)

    #Get small window
    radius = 16
    start = max(0, coarse_idx - radius)
    end = min(len(corr), coarse_idx + radius)

    window = corr[start:end]

    #Zero padded interpolation
    window_fft = np.fft.fft(window)

    #Zero pad
    n_original = len(window)
    n_padded = n_original * upsample_factor
    n_zeros = n_padded - n_original

    #FFT shift
    window_fft_shifted = np.fft.fftshift(window_fft)

    #insert zeros
    zeros = np.zeros(n_zeros, dtype=complex)
    fft_padded = np.concatenate([
            window_fft_shifted[:n_original//2], 
            zeros, 
            window_fft_shifted[n_original//2:]
        ])
    
    #IFFT
    fft_padded_ready = np.fft.ifftshift(fft_padded)
    window_upsampled= np.fft.ifft(fft_padded_ready) * upsample_factor

    #Find precise peak
    upsampled_mag = np.abs(window_upsampled)
    peak_upsampled_idx = np.argmax(upsampled_mag)

    #Calculate total delay
    fractional_offset = peak_upsampled_idx / upsample_factor

    total_idx = start + fractional_offset

    #Convert to time
    zero_lag_index = np.where(lags == 0)[0][0]
    final_lag_samples = total_idx - zero_lag_index

    return final_lag_samples / fs
    
if __name__ == "__main__":
    main()

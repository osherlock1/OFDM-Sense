import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
from ofdm.channel import delay
from ofdm.modulation import qam
from ofdm.config import OFDMConfig
from ofdm.core import sync
import scipy
import scipy.signal
import argparse
from ofdm.utils import usrp


#Config
CALIBRATION_PATH = "metadata/calibration.json"
RX_DATA_PATH = "data_files/rand_ofdm_packet.dat"
TX_REF_PATH = "data_files/rand_ofdm_packet_ref.json"
#Define OFDM Configuration
ofdm_conf = OFDMConfig()
STORE_REF_PATH = "./metadata/ref_delay_calc.json"
STORE_NO_REF_PATH = "./metadata/delay_calc.json"
USRP_CONFIG_PATH = "./configs/usrp_settings.yaml"
START_IDX_PATH = "./data_files/ofdm_performance.json"



C = scipy.constants.c #Speed of light

def calc_delay_with_ref():
    """
    Calculate delay between tx and rx using a wired references. 
    """
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

    json_data = {
        "delay":caled_delay,
        "raw_distance":raw_distance,
        "mode": "ref"
    }

    os.makedirs(os.path.dirname(STORE_REF_PATH), exist_ok=True)
    with open(STORE_REF_PATH, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Stored delay calc with ref to {STORE_REF_PATH}")




def calc_delay(rx_path:str, channel:str, start_idx:list, do_DebugPlot:bool = False):
    """
    Calculate delay for all cahnnels
    """
   #

    #Unpack Sivers RX and TX data
    raw_rx_data = np.fromfile(rx_path, dtype=np.complex64)
    coarse_sample_idx = start_idx[1]
    rx_data = clean_rx(raw_rx_data, coarse_sample_idx)

    #Unpack TX Pilot symbol
    with open("data_files/rand_ofdm_packet_ref.json", "r") as f:
        ref_data = json.load(f)

    #Get Tx pilot symbol
    tx_pilot = np.array(ref_data['pilot_ref_real']) + 1j * np.array(ref_data['pilot_ref_imag'])

    #Scale raw_rx_data
    scaled_wireless_rx = scale_rx_signal(raw_rx_data=rx_data)


    #Calculate Matched Filter Delay for wireless
    fine_delay_sec = calculate_precise_delay(
            rx_signal=scaled_wireless_rx,
            ref_signal=tx_pilot,
            fs=ofdm_conf.FS,
            debug_plot=do_DebugPlot
        )

    coarse_delay_sec = coarse_sample_idx / ofdm_conf.FS

    total_abosule_delay_sec = coarse_delay_sec + fine_delay_sec


    #Calculatiosn
    calced_delay_ns = ((total_abosule_delay_sec) * 1e9)

    print(f"Channel {channel}:")
    print(f"--Delay: {calced_delay_ns:.5f}ns")
    return calced_delay_ns
    



def clean_rx(rx_raw:np.ndarray, start_idx:int)->np.ndarray:
    """
    Removes leading and trialing zeros from the signal
    """

    # ------ Prepare RX signal --------
    with open(TX_REF_PATH, 'r') as f:
        ref_data = json.load(f)

    # Get total samples in packet
    sym_len = ofdm_conf.N + ofdm_conf.CP_LEN
    total_symbols = 1 + 1 + ref_data["n_data_symb"]
    total_samples = sym_len * total_symbols

    buffer = 0
    start = int(start_idx - buffer)
    end = int(start_idx + total_samples + buffer)

    if (start < 0): start = 0
    if (end > len(rx_raw)): end = len(rx_raw) - 1
    return rx_raw[start:end]
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", action="store_true", help = "Decalre if a directed wired ref is used")
    parser.add_argument("--plot", action="store_true", help ="Enable debug plot")
    parser.add_argument("--file", type=str, help="Specify specific file to unpack")
    args = parser.parse_args()

    with open(CALIBRATION_PATH, 'r') as f:
        cali_data = json.load(f)
    constants = cali_data['constants']

    with open(START_IDX_PATH, 'r') as f:
        ofdm_performance_data = json.load(f)

    start_idx_list = ofdm_performance_data['start_idx']
    
    calced_delays = []
    usrp_conf = usrp.load_config(USRP_CONFIG_PATH)
    rx_channel_idx = usrp_conf.rx_channel_idx.replace(",", "")

    if not args.file:
        for channel in rx_channel_idx:
            path = f"./data_files/rand_ofdm_packet_rx.0{channel}.dat"
            calced_delays.append(calc_delay(
                do_DebugPlot=args.plot,
                rx_path=path, 
                start_idx=start_idx_list[int(channel)],
                channel=channel))
    else:
        path = args.file
        calced_delays.append(calc_delay(
            do_DebugPlot=args.plot,
            rx_path=path, 
            start_idx=start_idx_list[0],
            channel=0
            ))


    json_data = {
        "delays":calced_delays,
        "mode": "no ref"
    }

    os.makedirs(os.path.dirname(STORE_NO_REF_PATH), exist_ok=True)
    with open(STORE_NO_REF_PATH, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Stored delay calc to {STORE_NO_REF_PATH}")

 
    

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


# def calculate_precise_delay(rx_signal, ref_signal, fs, upsample_factor=100, debug_plot:bool = False):
#     """
#     Calculates delay using coarse correlation zero-padded FFT interpolation
#     """
#     corr = scipy.signal.correlate(rx_signal, ref_signal, mode='full')
#     lags = scipy.signal.correlation_lags(len(rx_signal), len(ref_signal), mode='full')

#     # Find coarse peak
#     mag = np.abs(corr)
#     coarse_idx = np.argmax(mag)

#     #Get small window
#     radius = 16
#     start = max(0, coarse_idx - radius)
#     end = min(len(corr), coarse_idx + radius)

#     window = corr[start:end]

#     #Zero padded interpolation
#     window_fft = np.fft.fft(window)
#     window_fft_shifted = np.fft.fftshift(window_fft)

#     #Zero pad
#     n_original = len(window)
#     n_padded = n_original * upsample_factor
#     n_zeros = n_padded - n_original

#     zeros_half = np.zeros(n_zeros // 2, dtype=complex)
#     fft_padded_shifted = np.concatenate([
#             zeros_half, 
#             window_fft_shifted, 
#             zeros_half
#         ])
    
#     #IFFT
#     fft_padded_ready = np.fft.ifftshift(fft_padded_shifted)
#     window_upsampled= np.fft.ifft(fft_padded_ready) * upsample_factor

#     #Find precise peak
#     upsampled_mag = np.abs(window_upsampled)
#     peak_upsampled_idx = np.argmax(upsampled_mag)

#     #Calculate total delay
#     fractional_offset = peak_upsampled_idx / upsample_factor
#     total_idx = start + fractional_offset

#     #Convert to time
#     zero_lag_index = np.where(lags == 0)[0][0]
#     final_lag_samples = total_idx - zero_lag_index

#     if debug_plot:
#         plt.figure(figsize=(12, 5))
        
#         # Plot 1: The full coarse correlation
#         plt.subplot(1, 2, 1)
#         plt.plot(lags, mag)
#         plt.axvline(x=lags[coarse_idx], color='r', linestyle='--', alpha=0.7)
#         plt.title("Full Coarse Correlation")
#         plt.xlabel("Lags (Samples)")
#         plt.ylabel("Magnitude")

#         # Plot 2: The zoomed-in interpolated peak
#         plt.subplot(1, 2, 2)
#         # Create a sub-sample x-axis for the upsampled window
#         upsampled_lags = np.linspace(lags[start], lags[end-1], len(upsampled_mag))
#         plt.plot(upsampled_lags, upsampled_mag, label="Interpolated Curve")

#         plt.plot(lags[start:end], mag[start:end], 'ko', label="Coarse Samples")
        
#         precise_lag_val = lags[start] + fractional_offset
#         plt.axvline(x=precise_lag_val, color='r', linestyle='--', label=f"True Peak: {precise_lag_val:.2f}")
        
#         plt.title(f"Interpolated Peak (100x)")
#         plt.xlabel("Lags (Samples)")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()


#     return final_lag_samples / fs

def calculate_precise_delay(rx_signal, ref_signal, fs, debug_plot=False):
    """
    Calculates sub-sample delay using precise Parabolic Interpolation.
    """
    corr = scipy.signal.correlate(rx_signal, ref_signal, mode='full')
    lags = scipy.signal.correlation_lags(len(rx_signal), len(ref_signal), mode='full')

    mag = np.abs(corr)
    peak_idx = np.argmax(mag)

    # Parabolic Interpolation Formula
    if peak_idx > 0 and peak_idx < len(mag) - 1:
        alpha = mag[peak_idx - 1]
        beta  = mag[peak_idx]
        gamma = mag[peak_idx + 1]

        # Calculate the fractional shift (-0.5 to +0.5 samples)
        fractional_shift = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
    else:
        fractional_shift = 0.0

    precise_lag = lags[peak_idx] + fractional_shift

    if debug_plot:
        plt.figure(figsize=(8, 5))
        
        # Plot 10 samples on either side of the peak
        start = max(0, peak_idx - 10)
        end = min(len(mag), peak_idx + 10)
        
        plt.plot(lags[start:end], mag[start:end], 'ko-', label="Coarse Samples")
        plt.axvline(x=precise_lag, color='r', linestyle='--', label=f"Parabolic Peak: {precise_lag:.2f}")
        
        plt.title("Parabolic Interpolation Peak")
        plt.xlabel("Lags (Samples)")
        plt.ylabel("Correlation Magnitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return precise_lag / fs
    
if __name__ == "__main__":
    main()

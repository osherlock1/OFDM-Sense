import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
from ofdm.channel import delay
from ofdm.modulation import qam
from ofdm.config import OFDMConfig
import time
import datetime
import scipy

#Config
CALIBRATION_PATH = "metadata/calibration.json"
RX_DATA_PATH = "data_files/rand_ofdm_packet_rx.01.dat"
WIRED_DATA_PATH = "data_files/rand_ofdm_packet_rx.00.dat"
TX_REF_PATH = "data_files/rand_ofdm_packet_ref.json"

#C = scipy.constants.c #Speed of light
C = 299792458
REFERENCE_DISTANCE = 1 #1 Meter reference

def main():
    ofdm_conf = OFDMConfig()

    #Unpack Sivers RX-
    raw_rx_data = np.fromfile(RX_DATA_PATH, dtype=np.complex64)

    #Unpack Wired Ref RX
    wired_rx_data = np.fromfile(WIRED_DATA_PATH, dtype=np.complex64)

    #Unpack TX pilot symbol
    with open(TX_REF_PATH, 'r') as f:
        ref_data = json.load(f)
    
    #Get Tx Pilot symbol
    tx_pilot = np.array(ref_data['pilot_ref_real']) + 1j * np.array(ref_data['pilot_ref_imag'])

    #Scale raw_rx_data
    scaled_wireless_rx = scale_rx_signal(raw_rx_data=raw_rx_data)
    scaled_wired_rx = scale_rx_signal(raw_rx_data=wired_rx_data)

    #Upsample Data
    rx_wireless_upsampled = upsample(scaled_wireless_rx, scale_factor=100)
    rx_wired_upsampled = upsample(scaled_wired_rx, scale_factor=100)
    tx_upsampled = upsample(tx_pilot, scale_factor=100)

    #Calculate Matched Filter Delay for SIVERS
    z_wireless, lags_wireless = delay.matched_filter_calc(rx_iq = rx_wireless_upsampled, ref_iq = tx_upsampled, fs = (ofdm_conf.FS * 100))
    z_mag_wireless = np.abs(z_wireless)

    #Find Peak
    peak_idx_wireless = np.argmax(z_mag_wireless)
    fine_delay_wireless = lags_wireless[peak_idx_wireless]

    #Calculate Matched Filter Delay for WIRED
    z_wired, lags_wired = delay.matched_filter_calc(rx_iq = rx_wired_upsampled, ref_iq = tx_upsampled, fs = (ofdm_conf.FS * 100))
    z_mag_wired = np.abs(z_wired)

    #Find Peak
    peak_idx_wired = np.argmax(z_mag_wired)
    fine_delay_wired = lags_wireless[peak_idx_wired]

    #Calculate Calibration Constant
    constant = ((fine_delay_wireless - fine_delay_wired) * C) - REFERENCE_DISTANCE
    print(f"Calculated Constant: {constant}")

    #Save Constant to JSON
    json_data = {
        "reference_distance":REFERENCE_DISTANCE,
        "constant":constant,
        "calibration_time":datetime.datetime.now().isoformat()
    }

    os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"[Success] Calibration finished.  Saved constant to {CALIBRATION_PATH}")


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
    
def scale_rx_signal(raw_rx_data:np.ndarray)->np.ndarray:
    max_val = np.max(np.abs(raw_rx_data))

    if max_val > 0:
        scale_factor = 0.9 / max_val
        scaled_rx_data = raw_rx_data * scale_factor
    return scaled_rx_data

if __name__ == "__main__":
    main()
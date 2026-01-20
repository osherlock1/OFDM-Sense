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

#Config
CALIBRATION_PATH = "metadata/calibration.json"
RX_DATA_PATH = "data_files/rand_ofdm_packet.dat"
TX_REF_PATH = "data_files/rand_ofdm_packet_ref.json"

C = 299792458 #Speed of light
REFERENCE_DISTANCE = 1 #1 Meter reference

def main():
    ofdm_conf = OFDMConfig()

    #Unpack Raw RX and TX data
    raw_rx_data = np.fromfile(RX_DATA_PATH, dtype=np.complex64)

    #Unpack TX pilot symbol
    with open(TX_REF_PATH, 'r') as f:
        ref_data = json.load(f)
    
    #Get Tx Pilot symbol
    tx_pilot = np.array(ref_data['pilot_ref_real']) + 1j * np.array(ref_data['pilot_ref_imag'])

    #Scale raw_rx_data
    max_val = np.max(np.abs(raw_rx_data))

    if max_val > 0:
        scale_factor = 0.9 / max_val

    #Upsample Data
    rx_upsampled = upsample(raw_rx_data, scale_factor=100)
    tx_upsampled = upsample(tx_pilot, scale_factor=100)

    #Calculate Matched Filter Delay
    z, lags = delay.matched_filter_calc(rx_iq = rx_upsampled, ref_iq = tx_upsampled, fs = (ofdm_conf.FS * 100))
    z_mag = np.abs(z)

    #Find Peak
    peak_idx = np.argmax(z_mag)
    fine_delay = lags[peak_idx]

    #Calculate Calibration Constant
    constant = (fine_delay * C) - REFERENCE_DISTANCE
    print(f"Calculated Constant: {constant}")

    #Save Constant to JSON
    json_data = {
        "reference_distance":REFERENCE_DISTANCE,
        "constant":constant,
        "calibration_time":datetime.datetime.now().isoformat()
    }

    with open(CALIBRATION_PATH, "w") as f:
        json.dump(json_data)
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
    


if __name__ == "__main__":
    main()
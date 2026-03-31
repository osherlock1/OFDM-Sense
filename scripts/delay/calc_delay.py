import numpy as np
import os
import json
import scipy
import argparse

from ofdm.utils import usrp
from ofdm.modulation import qam
from ofdm.config import OFDMConfig
from ofdm.channel.delay import matched_filter_calc, scale_rx_signal, calculate_sub_sample_delay_parabolic, calculate_sub_sample_delay_zp

#Config
CALIBRATION_PATH = "metadata/calibration.json"
RX_DATA_PATH = "data_files/rand_ofdm_packet.dat"
TX_REF_PATH = "data_files/rand_ofdm_packet_ref.json"
#Define OFDM Configuration
OFDM_CONF = OFDMConfig()
STORE_REF_PATH = "./metadata/ref_delay_calc.json"
STORE_NO_REF_PATH = "./metadata/delay_calc.json"
USRP_CONFIG_PATH = "./configs/usrp_settings.yaml"
START_IDX_PATH = "./data_files/ofdm_performance.json"
C = scipy.constants.c #Speed of light

def clean_rx(rx_raw:np.ndarray, start_idx:int)->np.ndarray:
    """
    Removes leading and trialing zeros from the signal
    """
    with open(TX_REF_PATH, 'r') as f:
        ref_data = json.load(f)

    sym_len = OFDM_CONF.N + OFDM_CONF.CP_LEN
    total_symbols = 1 + 1 + ref_data["n_data_symb"]
    total_samples = sym_len * total_symbols

    buffer = 0
    start = int(start_idx - buffer)
    end = int(start_idx + total_samples + buffer)

    if (start < 0): start = 0
    if (end > len(rx_raw)): end = len(rx_raw) - 1
    return rx_raw[start:end]
    

def unpack_json(json_file_name:str)->dict:
    json_file_path = f"data_files/{json_file_name}"
    with open(json_file_path, "r") as f:
        data = json.load(f)
    print(f"[Success] Unpacked {json_file_path}")
    return data


def binary_ref_to_iq(binary_string:str, n_samples:int)->np.ndarray:
    full_string = "".join(binary_string)
    word_len = 4
    binary_word_list = np.array([full_string[i:word_len + i] for i in range(0 ,len(full_string), word_len)])
    iq_array = [qam.binary_to_iq(word) for word in binary_word_list]
    return np.array(iq_array) * np.sqrt(10)
     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", action="store_true", help = "Decalre if a directed wired ref is used")
    parser.add_argument("--plot", action="store_true", help ="Enable debug plot")
    parser.add_argument("--file", type=str, help="Specify specific file to unpack")
    args = parser.parse_args()

    usrp_conf = usrp.load_config(USRP_CONFIG_PATH)
    with open(CALIBRATION_PATH, 'r') as f:
        cali_data = json.load(f)
    constants = cali_data['constants']
    with open(START_IDX_PATH, 'r') as f:
        ofdm_performance_data = json.load(f)
    start_idx_list = ofdm_performance_data['start_idx']
    with open(TX_REF_PATH, "r") as f:
        ref_data = json.load(f)
    tx_pilot = np.array(ref_data['pilot_ref_real']) + 1j * np.array(ref_data['pilot_ref_imag'])

    calced_delays = []
    rx_channel_idx = usrp_conf.rx_channel_idx.replace(",", "")
    if not args.file:
        for channel in rx_channel_idx:
            path = f"./data_files/rand_ofdm_packet_rx.0{channel}.dat"
            rx_data = scale_rx_signal(clean_rx(np.fromfile(path, dtype=np.complex64), start_idx_list[int(channel)][1]))
            delay = calculate_sub_sample_delay_parabolic(rx_signal=rx_data, ref_signal=tx_pilot, fs = OFDM_CONF.FS) * 1e9 # convert to ns
            calced_delays.append(delay) 
            print(f"Channel{channel} delay: {delay:.1f}ns")
    else:
        path = args.file
        rx_data = scale_rx_signal(clean_rx(np.fromfile(path, dtype=np.complex64), start_idx_list[int(0)][1]))
        calced_delays.append(calculate_sub_sample_delay_parabolic(rx_signal=rx_data, ref_signal=tx_pilot, fs = OFDM_CONF.FS))

    json_data = {
        "delays":calced_delays,
        "mode": "no ref"
    }

    os.makedirs(os.path.dirname(STORE_NO_REF_PATH), exist_ok=True)
    with open(STORE_NO_REF_PATH, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Stored delay calc to {STORE_NO_REF_PATH}")

if __name__ == "__main__":
    main()

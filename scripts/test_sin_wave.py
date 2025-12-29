import numpy as np
import json
import os
import pathlib
import argparse
#Internal
from ofdm.utils import usrp


def main():
    parser = argparse.ArgumentParser(description="Generate Single Tone Sine wave for Testing Hardware")
    parser.add_argument('--channel', '-c', type=str, default = "A", help="Select Communication Channel(A = Baseband, B = Frontend)")
    parser.add_argument("--freq", type=float, default = 10e6, help = "Frequency of tone")
    parser.add_argument("--n_samps", type=int, default=2000, help = "Number of Samples sent")
    args = parser.parse_args()

    
    FS = 100e6
    FREQ = args.freq
    N_SAMLPES = args.n_samps
    CHANNEL = args.channel

    #Generate Test Tone
    generate_tone(fs=FS, freq=FREQ, n_samples= N_SAMLPES)

    #Define USRP Config
    usrp_conf = usrp.USRPConfig()
    
    #Send Data over USRP
    test_file_tx_path = "data_files/test_sin.dat"
    rx_file_path = "data_files/test_sin_rx.dat"
    usrp.run_transfer(channel=CHANNEL, config = usrp_conf, tx_file=test_file_tx_path, rx_file=rx_file_path, nsamps=N_SAMLPES)


def generate_tone(fs:float, freq:float, n_samples:float):

    t = np.arange(n_samples) / fs
    signal = np.exp(1j * 2 * np.pi * t)

    final_tx = signal

    #Save binary data for usrp
    bin_path = f"data_files/test_sin.dat"
    final_tx.astype(np.complex64).tofile(bin_path)
    print(f"[Success] Saved Binary Tone data to {bin_path}")

    #Save Refense Data
    referense_data = {
        "fs":fs,
        "freq":freq,
        "n_samples":n_samples,
        "singal_real":np.real(signal).tolist(),
        "signal_imag":np.imag(signal).tolist()
    }

    json_path =f"data_files/test_sin_ref.json"
    with open(json_path, "w") as f:
        json.dump(referense_data, f, indent=2)
    print(f"[Success] Saved Referense Sin Data to {json_path}")

if __name__ == "__main__":
    main()
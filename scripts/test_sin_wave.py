import numpy as np
import json
import os
import pathlib
import argparse
import matplotlib.pyplot as plt
#Internal
from ofdm.utils import usrp
from ofdm.viz import plotter

def main():
    parser = argparse.ArgumentParser(description="Generate Single Tone Sine wave for Testing Hardware")
    parser.add_argument('--channel', '-c', type=str, help="Define the subdev specifications (A:0 B:0 for both, A:0 for left daughter boards, B:0 for right daughter boards)")
    parser.add_argument("--freq", type=float, default = 10e6, help = "Frequency of tone")
    parser.add_argument("--n_samps", type=int, default=2000, help = "Specify number of samples in the generated tone")
    parser.add_argument("--tx_addr", type=str, help = "IP address of TX USRP (need to do in form of addr=(ip))")
    parser.add_argument("--rx_addr", type=str, help = "IP of RX USRP (need to do in form of addr=(ip))")
    parser.add_argument('--tx_channel_idx', type=str, help = "Select chanenl for the TX (based on subdev specification), if no subdev was specified default = A:0 B:0")
    parser.add_argument('--rx_channel_idx', type=str, help = "Select chanenl for the TX (based on subdev specification), if no subdev was specified default = A:0 B:0")
    parser.add_argument('--ref', type=str, help = "Specify the reference clock")
    args = parser.parse_args()

    
    FS = 100e6
    FREQ = args.freq
    N_SAMLPES = args.n_samps


    #Generate Test Tone
    BUFFER = 5000
    generate_tone(fs=FS, freq=FREQ, n_samples= N_SAMLPES, n_buffer = BUFFER)
    
    #Calculate number of samples to transfer
    nsamps_final = BUFFER * 2 + N_SAMLPES

    #Intantiate the USRP Config
    DEFAULT_CONFIG_PTH = "./configs/usrp_settings.yaml"
    usrp_conf = usrp.load_config(path=DEFAULT_CONFIG_PTH)

    #Overwrite configs with CLI args
    if args.channel is not None:
        usrp_conf.subdev = args.channel
        print(f"[USRP CONFIG] Updated subdev to:{args.channel}")

    if args.rx_addr is not None:
        usrp_conf.rx_addr = args.rx_addr
        print(f"[USRP CONFIG] Updated rx_addr to:{args.rx_addr}")

    if args.tx_addr is not None:
        usrp_conf.tx_addr = args.tx_addr
        print(f"[USRP CONFIG] Updated tx_addr to:{args.tx_addr}")
    
    if args.ref is not None:
        usrp_conf.ref = args.ref
        print(f"[USRP CONFIG] Updated ref to:{args.ref}")

    if args.tx_channel_idx is not None:
        usrp_conf.tx_channel_idx = args.tx_channel_idx
        print(f"[USRP CONFIG] Updated tx_channel_idx to:{args.tx_channel_idx}")
    
    if args.rx_channel_idx is not None:
        usrp_conf.rx_channel_idx = args.rx_channel_idx
        print(f"[USRP CONFIG] Updated rx_channel_idx {args.rx_channel_idx}")

    #Send Data over USRP
    test_file_tx_path = "data_files/test_sin.dat"
    rx_file_path = "data_files/test_sin_rx.dat"


    usrp.run_transfer(config = usrp_conf, tx_file=test_file_tx_path, rx_file=rx_file_path, nsamps=nsamps_final)

    #Unpack Rx Data
    print("[Test] Unpacking Data...")
    signal = np.fromfile(rx_file_path, dtype=np.complex64)

    #Unpack Ref Data
    with open("data_files/test_sin_ref.json", 'r') as f:
        data = json.load(f)
        ref_signal_real = np.array(data["signal_real"])
        ref_signal_imag = np.array(data["signal_imag"])
    ref_signal = ref_signal_real + 1j * ref_signal_imag

    #Calculate CFO
    T = 1 / FS
    freqs = np.fft.fftshift(np.fft.fftfreq(nsamps_final, T))

    rx_freq_idx = np.argmax(np.abs(np.fft.fftshift(np.fft.fft(signal))))
    rx_freq = freqs[rx_freq_idx]
    print(f"[Test] Calculated RX Freq: {rx_freq}Hz")
    print(f"[Test] Calculated CFO is {np.abs(rx_freq - FREQ)}")
    

    #Plot Rx data
    plotter.plot_time_series(signal = signal, fs = FS, title="Test Sine Wave Real")
    plotter.plot_symbol_freq(symbol = np.fft.fftshift(np.abs(np.fft.fft(signal))), title =f"Rx FFT Plot Freq = {FREQ}")

    #Plot Ref data
    plotter.plot_time_series(signal=ref_signal, fs = FS, title = "Ref Sine Wave Real")
    plotter.plot_symbol_freq(symbol = np.fft.fftshift(np.abs(np.fft.fft(ref_signal))), title="Ref FFT Plot")
    plt.show()

def generate_tone(fs:float, freq:float, n_samples:float, n_buffer: int = 1000):

    t = np.arange(n_samples) / fs
    signal = np.exp(1j * 2 * np.pi * freq * t)

    buffer = np.zeros(n_buffer, dtype=complex)

    final_tx = np.concatenate([buffer, signal, buffer])

    #Save binary data for usrp
    bin_path = f"data_files/test_sin.dat"
    final_tx.astype(np.complex64).tofile(bin_path)
    print(f"[Success] Saved Binary Tone data to {bin_path}")

    #Save Refense Data
    referense_data = {
        "fs":fs,
        "freq":freq,
        "n_samples":len(final_tx),
        "signal_real":np.real(final_tx).tolist(),
        "signal_imag":np.imag(final_tx).tolist()
    }

    json_path =f"data_files/test_sin_ref.json"
    with open(json_path, "w") as f:
        json.dump(referense_data, f, indent=2)
    print(f"[Success] Saved Referense Sin Data to {json_path}")

if __name__ == "__main__":
    main()
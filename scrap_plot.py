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
    FS = 100e6
    rx_file_path = "data_files/test_sin_rx.dat"

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
    # T = 1 / FS
    # freqs = np.fft.fftshift(np.fft.fftfreq(N_SAMLPES, T))

    # rx_freq_idx = np.argmax(np.abs(np.fft.fftshift(np.fft.fft(signal))))
    # rx_freq = freqs[rx_freq_idx]
    # print(f"[Test] Calculated RX Freq: {rx_freq}Hz")
    # print(f"[Test] Calculated CFO is {np.abs(rx_freq - FREQ)}")
    

    #Plot Rx data
    plotter.plot_time_series(signal = signal, fs = FS, title="Test Sine Wave Real")
    plotter.plot_symbol_freq(symbol = np.fft.fftshift(np.abs(np.fft.fft(signal))), title =f"Rx FFT Plot Freq =")

    #Plot Ref data
    plotter.plot_time_series(signal=ref_signal, fs = FS, title = "Ref Sine Wave Real")
    plotter.plot_symbol_freq(symbol = np.fft.fftshift(np.abs(np.fft.fft(ref_signal))), title="Ref FFT Plot")
    plt.show()


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
#internal
from ofdm.viz import plotter


def main():
    rx_file_path0 = "data_files/rand_ofdm_packet_rx.00.dat"
    rx_file_path1 = "data_files/rand_ofdm_packet_rx.01.dat"
    tx_file_path = "data_files/rand_ofdm_packet.dat"

    iq_data0 = np.fromfile(rx_file_path0, dtype=np.complex64)
    iq_data1 = np.fromfile(rx_file_path1, dtype=np.complex64)
    ref_data = np.fromfile(tx_file_path,dtype=np.complex64)

    #Plot TX and RX
    plotter.plot_time_series(signal=iq_data0, title="RX Data (Channel 0")
    plotter.plot_time_series(signal=iq_data0, title="RX Data1 (Channel 1)")
    plotter.plot_time_series(signal=ref_data, title="TX Ref Data")
    plt.show()

if __name__ == "__main__":
    main()
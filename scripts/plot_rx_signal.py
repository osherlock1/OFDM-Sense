import numpy as np
import matplotlib.pyplot as plt
#internal
from ofdm.viz import plotter
from ofdm.utils import usrp


def main():

    #Load Configurations
    USRP_CONFIG_PATH = "./configs/usrp_settings.yaml"
    usrp_conf = usrp.load_config(USRP_CONFIG_PATH)
    rx_channel_idx = usrp_conf.rx_channel_idx.replace(",", "")
    print(f"Unpacking {len(rx_channel_idx)} RX Channels...")


    for channel in rx_channel_idx:
        rx_file_path = f"data_files/rand_ofdm_packet_rx.0{channel}.dat"
        iq_data = np.fromfile(rx_file_path, dtype=np.complex64)

        #Plot TX and RX
        plotter.plot_time_series(signal=iq_data, title=f"RX Data (Channel {channel}")
    plt.show()

if __name__ == "__main__":
    main()
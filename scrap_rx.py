import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
#Internal
from ofdm.config import OFDMConfig
from ofdm.core import sync, waveform, preamble, payload
from ofdm.viz import plotter
from ofdm.channel import CHEST, cfo
from ofdm.utils import usrp
def main():
    ofdm_conf = OFDMConfig()
    usrp_conf = usrp.USRPConfig()

    ref_file_path = "data_files/rand_ofdm_packet_ref.json"

    with open(ref_file_path, "r") as f:
        ref_data = json.load(f)

    n_samps = ref_data['n_samples']


    tx_file_path = "data_files/rand_ofdm_packet.dat"

    rx_file_path = "data_files/rx_test.dat"

    usrp.run_rx(config = usrp_conf, rx_file= rx_file_path, channel= "A", nsamps=n_samps)
    print("Done")


if __name__ == "__main__":
    main()
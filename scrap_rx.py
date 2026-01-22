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
    usrp_conf = usrp.USRPConfig(rx_addr="addr=192.168.30.3")



    rx_file_path = "data_files/test_sin_rx.dat"

    usrp.run_rx(config = usrp_conf, rx_file= rx_file_path, channel= "A", nsamps=0)
    print("Done")


if __name__ == "__main__":
    main()
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

    tx_file_path = "data_files/rand_ofdm_packet.dat"

    usrp.run_tx(config = usrp_conf, tx_file= tx_file_path, channel= "A")
    print("Done")


if __name__ == "__main__":
    main()
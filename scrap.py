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
    usrp_conf = usrp.USRPConfig(tx_addr="addr=192.168.30.2")

    tx_file_path = "data_files/test_sin.dat"

    usrp.run_tx(config=usrp_conf, tx_file=tx_file_path, channel="B")

    print("Done")    




if __name__ == "__main__":
    main()
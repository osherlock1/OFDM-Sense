import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
#Internal
from ofdm.config import OFDMConfig

def main():
    parser = argparse.ArgumentParser(description="Unpack and Plot Recieved OFDM Packet")
    parser.add_argument('--file', type=str, default="rand_ofdm_packet_rx.dat", help="File name of packet to unpack")
    parser.add_argument('--ref', type=str, default ="rand_ofdm_packet_ref.json", help ="Reference packet json file name")
    args = parser.parse_args()

    #Load Configuration
    ofdm_conf = OFDMConfig()

    #Load Data
    print(f"Loading RX data from data_files/{args.file}...")
    rx_raw = np.fromfile(f"data_files/{args.file}")

    print(f"Loading Referense Data from data_files/{args.ref}...")
    with open(f"data_files/{args.ref}") as f:
        ref_data = json.load(f)

    n_payload_syms = ref_data["n_data_symbols"]

    # ---------- Syncronization -------------
    
if __name__ == "__main__":
    main()
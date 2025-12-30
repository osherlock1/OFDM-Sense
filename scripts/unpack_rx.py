import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
#Internal
from ofdm.config import OFDMConfig
from ofdm.core import sync

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
    
    #Unpack Referense Sync Symbol
    sync_ref_real = np.array(ref_data['sync_ref_real']).astype(complex)
    sync_ref_imag = np.array(ref_data["sync_ref_imag"]).astype(complex)
    sync_ref_time = sync_ref_real + 1j * sync_ref_imag

    n_payload_syms = ref_data["n_data_symbols"]

    # ---------- Syncronization -------------
    #Calc M and P Metrics
    M, P = sync.calculate_schmidl_cox_metrics(rx_signal=rx_raw, config=ofdm_conf)

    #Find Start Idx
    start_idx = sync.find_start_idx(
        M_metric=M,
        config=ofdm_conf,
        rx_signal=rx_raw,
        known_sync_time=sync_ref_time
    )

    #Estimate Coase CFO
    symbol_start_idx = start_idx + ofdm_conf.CP_LEN
    max_P = P[symbol_start_idx]
    coarse_cfo = sync.estimate_cfo_coarse(max_P, config=ofdm_conf)

    print(f"[Test] Coarse CFO:{coarse_cfo}, Start Idx:{start_idx}")



if __name__ == "__main__":
    main()
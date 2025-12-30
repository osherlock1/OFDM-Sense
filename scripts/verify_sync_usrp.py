import numpy as np
import json
import os
import pathlib
import argparse
import matplotlib.pyplot as plt
#Internal
from ofdm.config import OFDMConfig
from ofdm.utils import usrp
from ofdm.core import preamble
from ofdm.core import waveform
from ofdm.core import sync

def main():
    parser = argparse.ArgumentParser(description="Test Sync Function")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--channel", "-c", type=str, required=True, help="Select Communication Channel (A = Baseband, B = Frontend)")
    args = parser.parse_args()
    #Initialize OFDM Config
    ofdm_conf = OFDMConfig()

    #Generate Sync Symbol
    sync_freq = preamble.generate_sync_symbol(config=ofdm_conf, seed = args.seed)
    sync_tx = waveform.create_time_domain_symbol(freq_data=sync_freq, cp_len = ofdm_conf.CP_LEN)


    n_buffer = 300
    buffer = np.zeros(n_buffer, dtype=complex) #Zeros for start and end
    final_tx = np.concatenate([buffer, sync_tx, buffer])
    final_tx_ref = final_tx.copy
    
    #Save File
    sync_test_path_tx = "data_files/sync_verification_tx.dat"
    sync_test_ref_path = "data_files/sync_verification_ref.json"

    final_tx.astype(np.complex64).tofile(sync_test_path_tx)

    #Save Reference
    #Might not be necissary right now

    #Initiate USRP Config
    usrp_conf = usrp.USRPConfig()
    sync_test_path_rx = "data_files/sync_verification_rx"

    #Send Over USRP
    usrp.run_transfer(
        config=usrp_conf, 
        channel=args.channel, 
        tx_file=sync_test_path_tx,
        rx_file=sync_test_path_rx,
        nsamps=len(final_tx)
    )

    #Unpack Recieved Data
    print("[Test] Unpacking Data...")
    signal = np.fromfile(sync_test_path_rx, dtype=np.complex64)

    #Run Schmidl-Cox Sync
    M, P = sync.calculate_schmidl_cox_metrics(rx_signal=signal, config=ofdm_conf)

    #Get starting IDX
    peak_idx = sync.find_start_idx(
        M_metric=M,
        config=ofdm_conf, 
        rx_signal=signal, 
        known_sync_time=sync_tx,
        )

    #Calculate Coarse CFO
    max_P = P[peak_idx]
    cfo = sync.estimate_cfo_coarse(P_value=max_P, config= ofdm_conf)
    print(f"[Success! Estiamted Starting Idx:{peak_idx}, Estimated CFO:{cfo}Hz]")
    #Plot Results
    plt.figure()
    plt.plot(signal, label = "RX Signal")
    plt.plot(M, label = "M metric")
    plt.show()


    pass



if __name__ == "__main__":
    main()
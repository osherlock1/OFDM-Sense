import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
import shutil
from ofdm.channel import delay
from ofdm.modulation import qam
from ofdm.config import OFDMConfig
import scipy
import scipy.signal
import pandas as pd
import os
from datetime import datetime
import argparse
from ofdm.utils import usrp

EXPERIMENT_NAME = "trilat1_rx2_x50_8cm_y16_0cm"
EXPERIMENT_PATH = f"./experiments/{EXPERIMENT_NAME}.csv"

DELAY_DATA_PATH = "./metadata/delay_calc.json"
REF_DELAY_DATA_PATH = "./metadata/ref_delay_calc.json"
PERFORMANCE_DATA_PATH = "./data_files/ofdm_performance.json"
USRP_CONFIG_PATH = "./configs/usrp_settings.yaml"


SOURCE_DAT_FILE = "./data_files/rand_ofdm_packet_rx"

DESTINATION_DIR = f"/home/guoyixu/OFDM_Sense/EXPERIMENTS/{EXPERIMENT_NAME}_archive/"


#Load Configurations
USRP_CONFIG_PATH = "./configs/usrp_settings.yaml"
usrp_conf = usrp.load_config(USRP_CONFIG_PATH)
rx_channel_idx = usrp_conf.rx_channel_idx.replace(",", "")




def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--ref", action="store_true")
    parser.add_argument("--runs", type=int, default=1, help="specify number of time experiement will run")
    args = parser.parse_args()

    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
        print(f"Created archive directory: {DESTINATION_DIR}")

    for run in range(args.runs):

        print(f"\n========== RUN {run+1}/{args.runs} ==========\n")

        print("--- Starting Transfer (REF) ----")
        subprocess.run(["python", "./scripts/run_transfer.py"], check=True)
        print("--- Transfer Complete ---")
        
        print("---Unpacking OFDM Data---")
        subprocess.run(["python", "./scripts/unpack_rx.py"], check=True)
        print("---Finished Unpacking---")


        for channel in rx_channel_idx:

            source_dat_file = f"{SOURCE_DAT_FILE}.0{channel}.dat"
            channel_file = f"{DESTINATION_DIR}channel{channel}/"

            if not os.path.exists(channel_file):
                os.makedirs(channel_file)
                print(f"Created archive directory: {channel_file}")

            if os.path.exists(source_dat_file):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                new_filename = f"{EXPERIMENT_NAME}_run_{run+1}_{timestamp}.dat"
                destination_path = os.path.join(channel_file, new_filename)

                shutil.move(source_dat_file, destination_path)
                print(f"Saved run data to: {destination_path}")
            else:
                print(f"WARNING: Expected data file {source_dat_file} not found.")

if __name__ == "__main__":
    main()
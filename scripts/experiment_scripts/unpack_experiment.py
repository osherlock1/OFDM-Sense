import numpy as np
from pathlib import Path
import subprocess
import os
import json
import argparse
import pandas as pd
from ofdm.config import OFDMConfig
from ofdm.utils import usrp


DELAY_DATA_PATH = "./metadata/delay_calc.json"
PERFORMANCE_DATA_PATH = "./data_files/ofdm_performance.json"
USRP_CONFIG_PATH = "./configs/usrp_settings.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="path to raw .dat archive channel directory")
    parser.add_argument("--output_csv", type=str, required=True, help="path to output CSV file (e.g. ./experiments/my_experiment/rx3_channel1.csv)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    for dat_file in data_dir.glob("*.dat"):
        print(dat_file)

        print("---Unpacking OFDM Data---")
        subprocess.run(["python", "./scripts/unpack_rx.py", "--file", str(dat_file)], check=True)
        print("---Finished Unpacking---")

        print("---Calculating Delay -----")
        subprocess.run(["python", "./scripts/delay/calc_delay.py", "--file", str(dat_file)], check=True)

        with open(DELAY_DATA_PATH, 'r') as f:
            delay_data = json.load(f)

        with open(PERFORMANCE_DATA_PATH, 'r') as f:
            performance_data = json.load(f)

        evm = performance_data['evm']
        ber = performance_data['ber']
        ser = performance_data['ser']
        starting_idxs = performance_data['start_idx']
        delays = delay_data['delays']

        channel = 0
        results = {
            f'delay{channel}': delays[int(channel)],
            f'evm{channel}': evm[int(channel)],
            f'ber{channel}': ber[int(channel)],
            f'ser{channel}': ser[int(channel)],
            f'startingx_idx{channel}': starting_idxs[int(channel)][1]
        }

        df_new = pd.DataFrame([results])
        write_header = not os.path.exists(args.output_csv)
        df_new.to_csv(args.output_csv, mode='a', header=write_header, index=False)


if __name__ == "__main__":
    main()

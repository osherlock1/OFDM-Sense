import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
from ofdm.channel import delay
from ofdm.modulation import qam
from ofdm.config import OFDMConfig
import scipy
import scipy.signal
import pandas as pd
import os
from datetime import datetime
import argparse

EXPERIMENT_NAME = "test"
EXPERIMENT_PATH = f"./experiments/{EXPERIMENT_NAME}.csv"

DELAY_DATA_PATH = "./metadata/delay_calc.json"
REF_DELAY_DATA_PATH = "./metadata/ref_delay_calc.json"








def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--ref", action="store_true")

    args = parser.parse_args()




    if (args.ref == True):
        print("--- Starting Transfer (REF) ----")
        subprocess.run(["python", "./scripts/run_transfer.py"], check=True)
        print("--- Transfer Complete ---")

        print("---Calculating Delay -----")
        subprocess.run(["python", "./scripts/delay/calc_delay.py", "--ref"], check = True)
        #run subprocess to calculate delay

        with open(REF_DELAY_DATA_PATH, 'r') as f:
            delay_data = json.load(f)



        results = {
            'delay':delay_data["delay"],
            'distance':delay_data["raw_distance"]
        }

        df_new = pd.DataFrame([results])

        write_header = not os.path.exists(EXPERIMENT_PATH)

        df_new.to_csv(
            EXPERIMENT_PATH,
            mode = 'a',
            header=write_header,
            index=False
        )
    
    if (args.ref == False):
        print("--- Starting Transfer (NO REF) ----")
        subprocess.run(["python", "./scripts/run_transfer.py"], check=True)
        print("--- Transfer Complete ---")

        print("---Calculating Delay -----")
        subprocess.run(["python", "./scripts/delay/calc_delay.py"], check = True)
        #run subprocess to calculate delay

        with open(DELAY_DATA_PATH, 'r') as f:
            delay_data = json.load(f)

        delays = delay_data['delays']
        distances = delay_data['raw_distance']

        results = {
            'delay0':delays[0],
            'distance0':distances[0],
            'delay1':delays[1],
            'distance1':distances[1]
        }

        df_new = pd.DataFrame([results])

        write_header = not os.path.exists(EXPERIMENT_PATH)

        df_new.to_csv(
            EXPERIMENT_PATH,
            mode = 'a',
            header=write_header,
            index=False
        )




if __name__ == "__main__":
    main()
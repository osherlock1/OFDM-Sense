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

EXPERIMENT_NAME = "single_rx_no_ref"
EXPERIMENT_PATH = f"./experiments/{EXPERIMENT_NAME}.csv"


def main():
    

    print("--- Starting Transfer ----")
    subprocess.run(["python", "./scripts/run_transfer.py"], check=True)
    print("--- Transfer Complete ---")

    print("---Calculating Delay -----")
    subprocess.run(["python", "./scripts/calc_delay.py"], check = True)
    #run subprocess to calculate delay


if __name__ == "__main__":
    main()
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



def main():
    

    print("--- Starting Transfer ----")
    subprocess.run(["python", "./scripts/run_transfer.py"], check=True)
    print("--- Transfer Complete ---")

    print("---Calculating Delay -----")
    subprocess.run(["python", "./scripts/calc_delay.py"], check = True)
    #run subprocess to calculate delay


if __name__ == "__main__":
    main()
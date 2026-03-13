import numpy as np
from pathlib import Path
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
from ofdm.utils import usrp


data_dir = Path("./experiments/synthetic_trilateration1/trilat_rx3_x5_5in_y20_5in_archive/channel0")

for dat_file in data_dir.glob("*.dat"):
    
    file_path = data_dir / dat_file

    print("---Unpacking OFDM Data---")
    subprocess.run(["python", "./scripts/unpack_rx.py", "--file", str(dat_file)], check=True)
    print("---Finished Unpacking---")

    print("---Calculating Delay -----")
    subprocess.run(["python", "./scripts/delay/calc_delay.py"], check = True)
    #run subprocess to calculate delay
    break


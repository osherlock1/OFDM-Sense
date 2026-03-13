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


DELAY_DATA_PATH = "./metadata/delay_calc.json"
REF_DELAY_DATA_PATH = "./metadata/ref_delay_calc.json"
PERFORMANCE_DATA_PATH = "./data_files/ofdm_performance.json"
USRP_CONFIG_PATH = "./configs/usrp_settings.yaml"
UNPACKED_DATA_CSV = "./experiments/unpacked_data/rx3_channel1.csv"




data_dir = Path("/home/guoyixu/OFDM_Sense/EXPERIMENTS/synthetic_trilateration1/trilat_rx3_x5_5in_y20_5in_archive/channel1")

for dat_file in data_dir.glob("*.dat"):
    print(dat_file)
    file_path = data_dir / dat_file

    print("---Unpacking OFDM Data---")
    subprocess.run(["python", "./scripts/unpack_rx.py", "--file", str(dat_file)], check=True)
    print("---Finished Unpacking---")

    print("---Calculating Delay -----")
    subprocess.run(["python", "./scripts/delay/calc_delay.py", "--file", str(dat_file)], check = True)
    #run subprocess to calculate delay

    with open(DELAY_DATA_PATH, 'r') as f:
        delay_data = json.load(f)

    with open(PERFORMANCE_DATA_PATH, 'r') as f:
        performance_data = json.load(f)

    evm = performance_data['evm']
    ber = performance_data['ber']
    ser = performance_data['ser']
    starting_idxs = performance_data['start_idx']

    delays = delay_data['delays']

    usrp_conf = usrp.load_config(USRP_CONFIG_PATH)
    rx_channel_idx = usrp_conf.rx_channel_idx.replace(",", "")

    results = {}
    channel = 0
    results.update({
        f'delay{channel}':delays[int(channel)],
        f'evm{channel}': evm[int(channel)],
        f'ber{channel}':ber[int(channel)],
        f'ser{channel}':ser[int(channel)],
        f'startingx_idx{channel}' : starting_idxs[int(channel)][1]
    })

    df_new = pd.DataFrame([results])

    write_header = not os.path.exists(UNPACKED_DATA_CSV)

    df_new.to_csv(
        UNPACKED_DATA_CSV,
        mode = 'a',
        header=write_header,
        index=False
    )



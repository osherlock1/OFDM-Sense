from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
import numpy as np
from sync_symbol import SyncSymbol
import matplotlib.pyplot as plt
import random
import subprocess

import sys
values = np.linspace(1, 16, 16)
print(values)
scale_factor =[]
for value in values:
    scale_factor.append(str(round(value)))
print(scale_factor)


#scale_factor = "1"


for value in scale_factor:
    gen_cmd = [
        sys.executable,
        "pyscr/generate_ofdm_packet.py",
        "--snr", "100",
        "--seed", "42",
        "-s", value,
    ]

    run_cmd = [
        sys.executable,
        "pyscr/send_ofdm_file.py"
    ]


    result = subprocess.run(gen_cmd, check=True)
    send = subprocess.run(run_cmd, check=True)


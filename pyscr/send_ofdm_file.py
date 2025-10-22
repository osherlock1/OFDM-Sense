import subprocess, pathlib, shlex, sys, time
import numpy as np
import os, sys, numpy as np
import matplotlib.pyplot as plt
from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap

#CPP ARGS
BUILD_PATH = "./build/TXRX_TEST"
TX_ADDR = "addr=192.168.30.2"
RX_ADDR = "addr=192.168.30.2"
TX_RATE = "1e6"
RX_RATE = "1e6"
TX_FREQ = "0"
RX_FREQ = "0"
WAVE_TYPE = "SINE"
WAVE_FREQ = "100e3"
AMPL = "0.3"
TX_GAIN = "0"
RX_GAIN = "0"
OTW = "sc16"
TYPE = "float"
FILE_NAME = "data_files/usrp_samples_fc32_OUTPUT.dat"
NSAMPLES = "10000"
SETTLING = "0"
#TX_FILE = "data_files/usrp_samples_fc32_test.dat"
TX_FILE = "data_files/ofdm_iq_interleaved.dat"
TX_TYPE = "float"
TX_SPB = "0"
TX_REPEAT = "false"


#Build the command to run
run_cmd = [
    BUILD_PATH,
    "--tx-args", TX_ADDR,
    "--rx-args", RX_ADDR,
    "--tx-rate", TX_RATE,
    "--rx-rate", RX_RATE,
    "--tx-freq", TX_FREQ,
    "--rx-freq", RX_FREQ,
 #   "--wave-type", WAVE_TYPE,
 #   "--wave-freq", WAVE_FREQ,
 #   "--ampl", AMPL,
    "--tx-gain", TX_GAIN,
    "--rx-gain", RX_GAIN,
    "--otw", OTW,
    "--type", TYPE,
    "--file", FILE_NAME,
    "--nsamps", NSAMPLES,
    "--settling", SETTLING,
    "--tx-file", TX_FILE,
    "--tx-type", TX_TYPE,
    "--tx-spb", TX_SPB,
    "--tx-repeat", TX_REPEAT
]


# Run the CPP build
print("\n")
print(f"Running {BUILD_PATH}...")
print(str(run_cmd))
subprocess.run(run_cmd)
print(f"Run of {BUILD_PATH} complete!")


# ---------------------------------
# UNPACK OFDM SYMBOL
# --------------------------------
map = SubcarrierMap()
om = OFDMManager(map)


print("Unpacking OFDM Symbol...\n \n \n")
file_name = "data_files/usrp_samples_fc32_OUTPUT.dat"
file_size = os.path.getsize(file_name)
iq = np.fromfile(file_name, dtype = np.complex64)
plt.figure()
plt.plot(iq[:2000])
print("Calculating M Values... \n")
M_Values = []
for i in range(len(iq)):
    P, R, M = om.schmidl_cox_metrics_P_R_M(iq, delay=i)
    M_Values.append(M)
print("Done!\n")

plt.figure()
plt.plot(M_Values[:2000])
plt.show()



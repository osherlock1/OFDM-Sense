import subprocess, pathlib, shlex, sys, time
import numpy as np
import os, sys, numpy as np
import matplotlib.pyplot as plt


#CPP ARGS
BUILD_PATH = "./build/TXRX"
TX_ADDR = "addr=192.168.30.2"
RX_ADDR = "addr=192.168.30.2"
TX_RATE = "1e6"
RX_RATE = "1e6"
TX_FREQ = "0"
RX_FREQ = "0"
WAVE_TYPE = "SINE"
WAVE_FREQ = "500e3"
AMPL = "0.3"
TX_GAIN = "0"
RX_GAIN = "0"
OTW = "sc16"
TYPE = "float"
FILE_NAME = "usrp_samples_fc32.dat"
NSAMPLES = "1000000"
SETTLING = "0.2"

#Build the command to run
run_cmd = [
    BUILD_PATH,
    "--tx-args", TX_ADDR,
    "--rx-args", RX_ADDR,
    "--tx-rate", TX_RATE,
    "--rx-rate", RX_RATE,
    "--tx-freq", TX_FREQ,
    "--rx-freq", RX_FREQ,
    "--wave-type", WAVE_TYPE,
    "--wave-freq", WAVE_FREQ,
    "--ampl", AMPL,
    "--tx-gain", TX_GAIN,
    "--rx-gain", RX_GAIN,
    "--otw", OTW,
    "--type", TYPE,
    "--file", FILE_NAME,
    "--nsamps", NSAMPLES,
    "--settling", SETTLING
]


# Run the CPP build
print("\n")
print(f"Running {BUILD_PATH}...")
print(str(run_cmd))
subprocess.run(run_cmd)
print(f"Run of {BUILD_PATH} complete!")

print("Running python plotter")



file_name = "usrp_samples_fc32.dat"

#sample_rate = sys.argv[2]

file_size = os.path.getsize(file_name)

iq = np.fromfile(file_name, dtype = np.complex64)
print(file_size, "\n")
print(iq)


samples = np.fromfile(file_name, np.int16)
samples / 32768
samples = samples[::2] + 1j*samples[1::2]



sample_iq = iq[0:1000]
N = 1024
fft_iq = np.fft.fft(sample_iq, N)


magnitude = np.sqrt(np.real(sample_iq ** 2 + np.imag(sample_iq ** 2)))
print(f"simple_iq lenght {len(sample_iq)}")

plt.plot(np.abs(fft_iq))
plt.grid(True)
plt.title("100kHz Frequency Response")
plt.show()

plt.plot(np.real(sample_iq))
plt.grid(True)
plt.title("100kHz Sine Wave")
plt.show()

plt.plot(np.real(sample_iq), np.imag(sample_iq))
plt.grid(True)
plt.title("Constalation Plot")
plt.show()

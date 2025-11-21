import subprocess, pathlib, shlex, sys, time
import numpy as np
import os, sys, numpy as np
import matplotlib.pyplot as plt
from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
from data_generator import DataGenerator
import scipy
from scipy.interpolate import interp1d
import json
from pilot_symbol import PilotSymbol
import argparse
from scipy.signal import correlate





om = OFDMManager()
dg = DataGenerator(seed=43)
map = SubcarrierMap()
packet = dg.generate_random_packet(1)

data_k = map.data_bins
pilots_k = map.pilots_k

pilots_idx = np.array([map.idx(k) for k in pilots_k])

print(pilots_idx)
print(pilots_k)

data = packet[-64:]
pilot_data = data[pilots_idx]
print(pilot_data)


data_idx = np.array([map.idx(k) for k in data_k])

Tx = packet[-64:]
Tx_data = Tx[data_idx]

Tx_pilot = om.create_tx_block(PilotSymbol())
Tx_pilot = Tx_pilot[-64:]

Tx_data_pilot_freq = np.fft.fft(data, 64)
for i in range(map.N):
    if i not in pilots_idx:
        Tx_data_pilot_freq[i] = 0 + 0j

#print(Tx_data_pilot_freq)

cfo = 142e5
n_p = len(Tx_pilot)
n_p = np.arange(n_p)

fs = 100e6
bins = 2 ** 12
f_grid = np.linspace(-fs/2, fs/2, bins, endpoint = False)
n = len(Tx_data)
print(n)

n = np.arange(n)


Rx_pilot = Tx_pilot * np.exp(1j * 2*np.pi * cfo * n_p / fs)


Rx_data = Tx_data * np.exp(1j * 2*np.pi * cfo * n / fs)

frequencies = np.linspace(-fs, fs, bins)
#print(frequencies)


G, fhat= om.cfo_correct(Tx_data, Rx_data, fs)

plt.figure()
plt.plot(Tx_data, label="Tx data")
plt.plot(Rx_data, label="Rx Data")
plt.title("Tx vs Rx Simulated CFO")
plt.legend()


#print(G)
#Rx_pilot_corr = Rx_pilot * np.exp(-1j * 2*np.pi * f_hat * n_p / fs)
plt.figure()
plt.plot(f_grid, np.abs(G))
plt.xlabel("Frequency")
plt.title("|G(k)|" + "fs =" + str(fs / 1000000) + "MHz")
k = int(np.argmax(np.abs(G) ** 2))
#print(f"fhat is {f_hat}")
plt.show()

print(f"pilots k = {map.pilots_k}")
print(f"Number of Pilot Subcarrier = {len(map.pilots_k)}")
print(f"Calculated CFO = {fhat:.2f}")
print(f"Actual CFO = {cfo:.2f}")
print(f"Percent Difference = {(((np.abs(fhat - cfo)) / np.abs(cfo)) * 100):.2f}%")
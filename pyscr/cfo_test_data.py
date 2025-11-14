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

#Define Parameters
FS = 100e3
data_k = map.data_bins
pilot_k = map.pilots_k
pilot_idx = np.array([map.idx(k) for k in pilot_k])
data_idx = np.array([map.idx(k) for k in data_k])
CFO = 2343
n = np.arange(map.N)
f_grid = np.linspace(-FS/2, FS/2, 2 ** 12)
print(n)

#Unpack Data Symbol
txn = packet[-64:]

#Apply CFO to Received Signal
rxn = txn * np.exp(-1j * 2*np.pi * CFO * n / FS)

plt.figure()
plt.plot(txn)
plt.plot(rxn)
plt.title("Origianl Data Symbols Time Domain")
plt.show()


#Convert txn and rxn to freq domain
TXk = np.fft.fft(txn, map.N)
RXk = np.fft.fft(rxn, map.N)

#Zero out pilot carrier
TXk_pilot = TXk
for i in range(map.N):
    if i not in pilot_idx:
        TXk_pilot[i] = 0 + 0j




RXk_pilot = RXk
for i in range(map.N):
    if i not in pilot_idx:
        RXk_pilot[i] = 0 + 0j
plt.figure()
plt.plot(np.abs(TXk_pilot))
plt.plot(np.abs(RXk_pilot))
plt.title("FTT result")
plt.show()

# for k in pilot_k:
#     print("TX \n")
#     print(TXk_pilot[map.idx(k)])
#     print("RX \n")
#     print(RXk_pilot[map.idx(k)])

#Go back to time domain
txn_time = np.fft.ifft(TXk_pilot, 64)
rxn_time = np.fft.ifft(RXk_pilot, 64)
    
plt.figure()
plt.plot(txn_time)
plt.plot(rxn_time)
plt.title("Data Symbol Pilot Subcarriers Only IFFT Result")
plt.show()
#Calculate G
G, f_hat = om.cfo_correct(Tx = txn_time, Rx = rxn_time, fs = FS)

print(f_hat)

#print(laps)
plt.figure()
plt.plot(f_grid, np.abs(G))
plt.show()
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
dg = DataGenerator(seed=42)
map = SubcarrierMap()
packet = dg.generate_random_packet(1)

data_k = map.data_bins

data_idx = np.array([map.idx(k) for k in data_k])

Tx = packet[-64:]
Tx_data = Tx[data_idx]
cfo = 12327
Rx_data = Tx_data * ((np.e ** (-1j * cfo)))


correlation = correlate((Tx_data), np.conj((Rx_data)), mode='full')
correlation /= np.sqrt(np.dot(Tx_data, Tx_data) * (np.dot(Rx_data, Rx_data)))

corr_sum = np.sum(correlation)
print(np.abs(corr_sum))

"""

fs = 100e3




frequency_bins = np.linspace(-fs, fs, len(Tx_data))
print(len(frequency_bins))

G_vector = om.cfo_adjustment(Tx_data, Rx_data, fs)
print(len(G_vector))


plt.figure()
plt.plot(frequency_bins, np.abs(G_vector))

plt.figure()
plt.plot(Rx_data)
plt.plot(Tx_data)

plt.show()
"""
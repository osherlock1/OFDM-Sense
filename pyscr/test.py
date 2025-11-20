from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
import numpy as np
from sync_symbol import SyncSymbol
import matplotlib.pyplot as plt
import random
import subprocess
from data_generator import DataGenerator
import sys


om = OFDMManager()
dg =DataGenerator()
map = SubcarrierMap()


rand_binary = dg.generate_random_binary(len(map.data_bins) * 4)

parsed_binary = dg._parse_string(rand_binary, 4)

iq_samples = np.array([om.binary_to_iq(bin) for bin in parsed_binary], dtype = np.complex128)

pilots = np.ones(12, dtype=np.complex128)  * 3 / (np.sqrt(10))


ofdm_packet = dg.generate_random_packet(N_data_symbols=1)

datasymbol = ofdm_packet[-64:]
data_fft = np.fft.fft(datasymbol)
print(data_fft[map.pilots_k])




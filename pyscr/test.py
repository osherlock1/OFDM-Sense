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
from pilot_symbol import PilotSymbol

om = OFDMManager()
dg =DataGenerator()
map = SubcarrierMap()
cp_len = map.cp_len

pilot_ref = om.create_tx_block(PilotSymbol())
data_k = map.data_bins
data_idx = np.array([map.idx(k) for k in data_k])
pilot_rx = pilot_ref[map.cp_len - 4: -4] * 0.7
pilot_tx = pilot_ref[map.cp_len:]


pilot_RK = np.fft.fft(pilot_rx)
pilot_TK = np.fft.fft(pilot_tx)

data_rn = dg.generate_random_packet(N_data_symbols = 1)

data_ref_n = data_rn[-64:]
data_rx_n = data_rn[-68:-4]
print(len(data_ref_n))
print(len(data_rx_n))
data_rx_K = np.fft.fft(data_rx_n)
data_ref_K = np.fft.fft(data_ref_n)


plt.figure()
plt.plot(pilot_RK)
plt.plot(pilot_TK)
#plt.show()



def channel_estimation(recieved_pilot_symbol:np.ndarray, known_pilot_symbol:np.ndarray):
    
    data_k = map.data_bins
    data_idx = np.array([map.idx(k) for k in data_k])
    
    
    r = recieved_pilot_symbol[data_idx]
    s = known_pilot_symbol[data_idx]
    s_conj = np.conj(s)
    sqr_mag_s = np.abs(s) ** 2
    channel_gain = (r * s_conj) / sqr_mag_s
    return channel_gain


lambda_k = channel_estimation(recieved_pilot_symbol=pilot_RK, known_pilot_symbol=pilot_TK)

plt.figure()
plt.plot(np.angle(lambda_k))
plt.show()

Y = pilot_RK[data_idx] / lambda_k
yk = data_rx_K[data_idx] / lambda_k
plt.figure()
plt.plot(Y, label = "Y")
plt.plot(pilot_TK[data_idx])
plt.plot(pilot_RK[data_idx])
plt.legend()
plt.show()

plt.figure()
plt.plot(yk, label = "yk")
plt.plot(data_ref_K[data_idx])
plt.show()


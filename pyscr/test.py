from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
import numpy as np
from sync_symbol import SyncSymbol
import matplotlib.pyplot as plt


map = SubcarrierMap()
iq_samples = np.array([-0.9487 + 0.9487j, -0.3162 + 0.9487j,  0.3162 + 0.9487j,  0.9487 + 0.9487j,
                        -0.9487 + 0.3162j, -0.3162 + 0.3162j,  0.3162 + 0.3162j,  0.9487 + 0.3162j,
                        -0.9487 - 0.3162j, -0.3162 - 0.3162j,  0.3162 - 0.3162j,  0.9487 - 0.3162j,
                        -0.9487 - 0.9487j, -0.3162 - 0.9487j,  0.3162 - 0.9487j,  0.9487 - 0.9487j,

                        -0.9487 + 0.9487j, -0.3162 + 0.9487j,  0.3162 + 0.9487j,  0.9487 + 0.9487j,
                        -0.9487 + 0.3162j, -0.3162 + 0.3162j,  0.3162 + 0.3162j,  0.9487 + 0.3162j,
                        -0.9487 - 0.3162j, -0.3162 - 0.3162j,  0.3162 - 0.3162j,  0.9487 - 0.3162j,
                        -0.9487 - 0.9487j, -0.3162 - 0.9487j,  0.3162 - 0.9487j,  0.9487 - 0.9487j,

                        -0.9487 + 0.9487j, -0.3162 + 0.9487j,  0.3162 + 0.9487j,  0.9487 + 0.9487j,
                        -0.9487 + 0.3162j, -0.3162 + 0.3162j,  0.3162 + 0.3162j,  0.9487 + 0.3162j,
                        -0.9487 - 0.3162j, -0.3162 - 0.3162j,  0.3162 - 0.3162j,  0.9487 - 0.3162j,
                        -0.9487 - 0.9487j, -0.3162 - 0.9487j,  0.3162 - 0.9487j,  0.9487 - 0.9487j], dtype=complex)


pilots = np.array([1 + 0j, 1 + 0j, 1+0j, 1+0j], dtype=complex)


om = OFDMManager(SubcarrierMap())

data_symbol = OFDMSymbol(iq_samples48=iq_samples, pilots4=pilots, submap=map)
tx_data_block = om.create_tx_block(data_symbol)
sync = SyncSymbol()
sync_symbol = SyncSymbol()
tx_block = om.create_tx_block(sync)

send_signal = np.concat([tx_data_block, tx_block, tx_data_block, tx_data_block, tx_data_block])
P_values = []
M_values = []
R_values = []
for i in range(150):
    P, R, M = om.schmidl_cox_metrics(r = send_signal, delay = i)
    P_values.append(P)
    R_values.append(R)
    M_values.append(M)

print(np.abs(M_values))

plt.figure()
plt.plot(np.abs(P_values))
plt.title("P Values")

plt.figure()
plt.plot(R_values)
plt.title("R Values")

plt.figure()
plt.plot(M_values)
plt.title("M values")
plt.show()
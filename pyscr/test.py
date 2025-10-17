from ofdm_manager import OFDMManager
from subcarrier_map import SubcarrierMap
import numpy as np
from sync_symbol import SyncSymbol
import matplotlib.pyplot as plt


om = OFDMManager(SubcarrierMap())

sync = SyncSymbol()
om.ifft(sync)


plt.figure()
plt.plot(np.real(sync.symbol))
plt.show()



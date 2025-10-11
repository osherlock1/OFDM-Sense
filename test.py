import numpy as np

fname = "usrp_samples.dat"

iq = np.fromfile(fname, np.int16)
iq = iq.reshape(-1, 2).astype(np.float32)
x = iq[:,0] + 1j*iq[:,1]
#x /= 32768.0
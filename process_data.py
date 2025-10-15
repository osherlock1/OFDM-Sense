import os, sys, numpy as np
import matplotlib.pyplot as plt



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
plt.show()



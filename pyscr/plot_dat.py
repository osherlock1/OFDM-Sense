import numpy as np, matplotlib.pyplot as plt, sys, os

file_name = sys.argv[1]
Fs = float(sys.argv[2])

data = np.fromfile(file_name, np.complex64)
print(f"Samples: {len(data)}, file size: {os.path.getsize(file_name)} bytes")

# Look at raw stats
print("min:", data.min(), "max:", data.max(), "mean:", np.mean(np.abs(data)))

# Plot first few samples
plt.plot(np.real(data[:2000]), label="I")
plt.plot(np.imag(data[:2000]), label="Q")
plt.legend(); plt.title("First 2000 samples"); plt.grid(); plt.show()

# Spectrum check
plt.figure()
plt.magnitude_spectrum(data[:131072], Fs=Fs)
plt.title("Magnitude Spectrum"); plt.show()

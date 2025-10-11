import os, sys, numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python3 plot_dat.py <usrp_samples.dat> <fs_in_sps>")
    sys.exit(1)

fname = sys.argv[1]
fs = float(sys.argv[2])  # sample rate you used, e.g. 1e6

size = os.path.getsize(fname)

def try_load(dtype, itemsize):
    # dtype pair stream -> complex
    raw = np.fromfile(fname, dtype=dtype)
    if raw.size % 2 != 0:
        return None
    iq = np.fromfile(fname, np.int16)
    iq = iq.reshape(-1, 2).astype(np.float32)
    x = iq[:,0] + 1j*iq[:,1]
    return x

# heuristic: check both
x_fc32 = try_load(np.float32, 8)
x_sc16 = try_load(np.int16,   4)

# pick one by file-size divisibility and dynamic range
pick = None
if size % 8 == 0 and x_fc32 is not None:
    pick = ('fc32', x_fc32)
elif size % 4 == 0 and x_sc16 is not None:
    pick = ('sc16', x_sc16)
else:
    print("Could not determine format; file size not compatible with fc32 or sc16.")
    sys.exit(2)

fmt, x = pick
print(f"Loaded {fname} as {fmt}, {x.size} complex samples")

# If sc16, normalize to [-1,1] range for viewing
if fmt == 'sc16':
    x = x / 32768.0

# --- Time domain (first 5 ms)
N_show = min(len(x), int(0.005*fs))
t = np.arange(N_show)/fs
plt.figure()
plt.plot(t, np.real(x[:N_show]))
plt.title("I (time domain)")
plt.xlabel("Time [s]"); plt.ylabel("Amplitude"); plt.tight_layout()

# --- Spectrum (Welch PSD-ish)
NFFT = 16384 if len(x) >= 16384 else 2**int(np.floor(np.log2(len(x))))
w = np.hanning(NFFT)
X = np.fft.fftshift(np.fft.fft(w * x[:NFFT], NFFT))
f = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1/fs))
psd = 20*np.log10(np.maximum(np.abs(X), 1e-12))

plt.figure()
plt.plot(f/1e3, psd)
plt.title("Magnitude Spectrum")
plt.xlabel("Frequency [kHz]"); plt.ylabel("Mag [dB]"); plt.tight_layout()

# --- Constellation (if it’s a tone you’ll see a circle/point)
Ns = min(len(x), 20000)
plt.figure()
plt.plot(np.real(x[:Ns]), np.imag(x[:Ns]), '.', markersize=1)
plt.title("Constellation (first 20k samples)")
plt.xlabel("I"); plt.ylabel("Q"); plt.axis('equal'); plt.tight_layout()

plt.show()

import numpy as np
import uhd
import matplotlib.pyplot as plt
import time

USRP_ARGS = "type=x300,addr=192.168.30.2"
RATE = 1e6
CENTER_HZ = 915e6
GAIN_DB = 20
N_SAMPS = 200000
CHAN = 0
ANT = "A"

usrp = uhd.usrp.MultiUSRP(USRP_ARGS)
usrp.set_rx_rate(RATE, CHAN)
usrp.set_rx_freq(uhd.types.TuneRequest((CENTER_HZ), CHAN))
usrp.set_rx_gain(GAIN_DB, CHAN)
usrp.set_rx_antenna(ANT, CHAN)

stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [CHAN]
rx_stream = usrp.get_rx_stream(stream_args)
buff = np.zeros((N_SAMPS,), dtype = np.complex64)

# Timed start a bit in the future to align with your TX if you want
time.sleep(0.1)
md = uhd.types.RXMetadata()
rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.num_done))
rx_stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
rx_stream_cmd.num_samps = N_SAMPS
rx_stream_cmd.stream_now = True
rx_stream.issue_stream_cmd(rx_stream_cmd)

num_accum = 0
while num_accum < N_SAMPS:
    samps = rx_stream.recv(buff[num_accum:], md, timeout=1.0)
    if md.error_code != uhd.types.RXMetadataErrorCode.none:
        raise RuntimeError(f"RX error: {md.strerror()}")
    num_accum += samps

# ---- Plots ----
fs   = RATE
t    = np.arange(N_SAMPS)/fs
x    = buff

# Time-domain (I/Q magnitude)
plt.figure()
plt.plot(t[:4000], np.real(x[:4000]))      # quick peek
plt.title("Time domain (I)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.tight_layout()

# Spectrum
NFFT = 16384
Xf   = np.fft.fftshift(np.fft.fft(x[:NFFT]*np.hanning(NFFT), NFFT))
f    = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1/fs))
psd  = 20*np.log10(np.maximum(np.abs(Xf), 1e-12))

plt.figure()
plt.plot(f/1e3, psd)
plt.title("Spectrum")
plt.xlabel("Frequency offset from LO [kHz]")
plt.ylabel("Magnitude [dB]")
plt.tight_layout()

# Spectrogram (coarse)
plt.figure()
plt.specgram(x, NFFT=2048, Fs=fs, noverlap=1024)
plt.title("Spectrogram")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.tight_layout()

plt.show()

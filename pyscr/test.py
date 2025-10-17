import numpy as np
import matplotlib.pyplot as plt

symbol = -0.9487 + 0.9487j

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

# Generate a complex sinusoidal wave
fs = int(1e5)  # Sample rate (samples per second)
f = 1e4       # Frequency in Hz (10 kHz)
duration = 0.001  # Duration in seconds (1 ms)
num_samples = int(fs * duration)

# Create time array
t = np.arange(num_samples) / fs

# Initialize the final wave array
final_wave = np.zeros(num_samples, dtype=complex)

# Generate and sum modulated waves for each IQ sample
for sample in iq_samples:
    # Generate carrier wave and modulate it with the IQ sample
    carrier = np.exp(1j * 2 * np.pi * f * t)
    modulated_wave = sample * carrier
    final_wave = final_wave + modulated_wave

# Compute FFT of the final summed wave
freq = np.fft.fft(final_wave, 1024)

# Plot the results
plt.figure(figsize=(12, 8))

# Plot time domain - real part
plt.subplot(2, 2, 1)
plt.plot(t * 1000, np.real(final_wave))
plt.title('Time Domain - Real Part')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot time domain - imaginary part
plt.subplot(2, 2, 2)
plt.plot(t * 1000, np.imag(final_wave))
plt.title('Time Domain - Imaginary Part')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot frequency domain - magnitude
plt.subplot(2, 2, 3)
freqs = np.fft.fftfreq(1024, 1/fs)
plt.plot(freqs[:512], np.abs(freq[:512]))  # Plot positive frequencies only
plt.title('Frequency Domain - Magnitude')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot constellation (IQ samples)
plt.subplot(2, 2, 4)
plt.scatter(np.real(iq_samples), np.imag(iq_samples), alpha=0.7)
plt.title('Constellation Diagram')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()

print(f"Summed {len(iq_samples)} modulated waves")
print(f"Final wave has {len(final_wave)} samples")
print(f"Peak amplitude: {np.max(np.abs(final_wave)):.3f}")
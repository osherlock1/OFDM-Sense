import matplotlib.pyplot as plt
import numpy as np
from subcarrier_map import SubcarrierMap
from ofdm_symbol import OFDMSymbol
from ofdm_manager import OFDMManager

result = 3 / np.sqrt(10)
print(result)
    

def main():


    map = SubcarrierMap()
    om = OFDMManager(map)

     #Simple IQ samples respoinse
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

    ofdm_symbol1 = OFDMSymbol(iq_samples48=iq_samples, pilots4=pilots, submap=map)
    ofdm_symbol2 = OFDMSymbol(iq_samples48=iq_samples, pilots4=pilots, submap=map)
    TX_Block2 = om.create_tx_block(ofdm_symbol2)
    om.ifft(ofdm_symbol1)
    fs = int(1e5)  # Sample rate (samples per second)
    f = 1e4       # Frequency in Hz (10 kHz)
    duration = 0.001  # Duration in seconds (1 ms)
    num_samples = int(fs * duration)

    # Create time array
    t = np.arange(num_samples) / fs



    # Initialize the final wave array
    final_wave = np.zeros(num_samples, dtype=complex)
    fft_list = []
    # Generate and sum modulated waves for each IQ sample
    for sample in ofdm_symbol1.symbol:
        # Generate carrier wave and modulate it with the IQ sample
        carrier = np.exp(1j * 2 * np.pi * f * t)
        f = f + 1000
        modulated_wave = sample * carrier
        fft_result = np.fft.fft(modulated_wave, 1024)
        fft_list.append(fft_result)
        final_wave = final_wave + modulated_wave


    plt.figure()
    for result in fft_list:
        plt.plot(np.real(result))
    plt.title("final wave")


    #Plot the grey coded QAM-16 map 
    plt.figure()
    plt.plot(np.real(iq_samples), np.imag(iq_samples), '.')
    plt.title("QAM-16 Grey-coded normalized Map")


    plt.figure()
    plt.plot(np.real(TX_Block2))
    plt.plot(np.imag(TX_Block2))
    plt.title("Final OFDM Symbol + CP")
    plt.legend()
    plt.figure()
    plt.plot(TX_Block2)
    plt.title("TX Block")
    plt.show()

if __name__ == "__main__":
    main()



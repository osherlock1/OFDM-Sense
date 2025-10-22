import numpy as np
from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
import matplotlib.pyplot as plt
from data_generator import DataGenerator
import random

def main():
    
    #Instantaite objects
    map = SubcarrierMap()
    om = OFDMManager()
    data_gen = DataGenerator()

    #Genearte random OFDM pakcet
    packet = data_gen.generate_random_packet(N_data_symbols= 6)

    #Add Noise Buffer
    noise_buffer = generate_noise_buffer(100)
    #Concat noise buffer to packet
    packet = np.concatenate([noise_buffer, packet, noise_buffer])
    

    #Calculate M Values
    M_values = []
    for delay_i in range(int(len(packet))):
        P, R, M = om.schmidl_cox_metrics_P_R_M(r = packet, delay=delay_i)
        M_values.append(M)

    #Plot the OFDM packet
    plt.figure()
    plt.plot(packet)
    plt.plot(M_values)
    plt.show()


def generate_noise_buffer(buf_len:int = 200) -> np.ndarray:
    noise_buffer = []
    for i in range(buf_len):
        noise_buffer.append(random.uniform(-0.3,0.3))
    return np.array(noise_buffer, dtype=complex)

if __name__ == "__main__":
    main()

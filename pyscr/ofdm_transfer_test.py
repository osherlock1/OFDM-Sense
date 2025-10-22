import numpy as np
from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
import matplotlib.pyplot as plt
from data_generator import DataGenerator
import random
import scipy

def main():
    
    #Instantaite objects
    map = SubcarrierMap()
    om = OFDMManager()
    data_gen = DataGenerator()

    #Genearte random OFDM pakcet
    N_data_symbols = 6
    N_symbols = N_data_symbols + 1
    packet = data_gen.generate_random_packet(N_data_symbols= N_data_symbols)

    #Add Noise Buffer
    noise_buffer = generate_noise_buffer(100)
    #Concat noise buffer to packet
    packet = np.concatenate([noise_buffer, packet, noise_buffer])
    

    #Calculate M Values
    M_values = []
    for delay_i in range(int(len(packet))):
        P, R, M = om.schmidl_cox_metrics_P_R_M(r = packet, delay=delay_i)
        M_values.append(M)

    #Filter M values
    ofdm_CP = 8
    b_toPeak = np.ones(ofdm_CP) / ofdm_CP
    a = (1,)
    M_filter = scipy.signal.lfilter(b_toPeak, a, M_values)
    
    #Calculate derivative of filtered M
    D = np.diff(M_filter)

    #Find zero crossings
    zeroCrossing_1 = (D[:1] * D[1:]) <= 0
    M_values_np = np.array(M_values, dtype=complex)
    zeroCrossing_2 = ((D[:-1] * D[1:]) <= 0) * (M_values_np[1:-1] > 0.5)

    #Remove duplicate zero crossings
    b_preamble = np.zeros(map.N + N_symbols * (map.N + ofdm_CP))
    b_preamble[:map.N] = 1

    b_payload = np.zeros(map.N + N_symbols*(map.N + ofdm_CP))
    for s in range(N_data_symbols):
        b_payload[map.N + (s + 1)*ofdm_CP + s * map.N + np.arange(map.N)] = 1

    preamble_valid_est = scipy.signal.lfilter(b_preamble, (1,), zeroCrossing_2)
    payload_valid_est = scipy.signal.lfilter(b_payload, (1,), zeroCrossing_2)

    #Plot the OFDM packet
    plt.figure()
    plt.plot(packet, label = "OFDM Packet")
    plt.plot(M_values, label = "M Values")
    plt.plot(M_filter, label = "Filtered M")
    plt.plot(D, label = "Derivative of M_filter")
    plt.plot(zeroCrossing_2, label = "zero crossings")
    plt.plot(preamble_valid_est, label = "Estiamtion of valid Sync")
    plt.plot(payload_valid_est, label = "Estimation of valid payload")
    plt.legend()
    plt.show()


def generate_noise_buffer(buf_len:int = 200) -> np.ndarray:
    noise_buffer = []
    for i in range(buf_len):
        noise_buffer.append(random.uniform(-0.1,0.1))
    return np.array(noise_buffer, dtype=complex)

if __name__ == "__main__":
    main()

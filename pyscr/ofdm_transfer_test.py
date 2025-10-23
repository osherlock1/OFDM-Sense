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

    #data from file
    try:
        file_iq = np.fromfile("data_files/usrp_samples_fc32_OUTPUT.dat", dtype=np.complex64)
    except:
        return ValueError(
            print("File does not exist")
        )

    #file_iq = normalize_samples(file_iq)
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
    for delay_i in range(int(len(file_iq))):
        P, R, M = om.schmidl_cox_metrics_P_R_M(r = file_iq, delay=delay_i)
        M_values.append(M)

    plt.figure()
    plt.plot(M_values)
    plt.plot(file_iq)
    plt.show()



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

    b_ignore = np.ones(1+map.N)
    b_ignore[0] = 0
    ignore_times = (scipy.signal.lfilter(b_ignore, (1, ), zeroCrossing_2) > 0).astype(int)

    zeroCrossing_3 = zeroCrossing_2 * (ignore_times == 0)

    #Remove duplicate zero crossings
    b_preamble = np.zeros(map.N + N_symbols * (map.N + ofdm_CP))
    b_preamble[:map.N] = 1

    b_payload = np.zeros(map.N + N_symbols*(map.N + ofdm_CP))
    for s in range(N_data_symbols):
        b_payload[map.N + (s + 1)*ofdm_CP + s * map.N + np.arange(map.N)] = 1

    preamble_valid_est = scipy.signal.lfilter(b_preamble, (1,), zeroCrossing_3)
    payload_valid_est = scipy.signal.lfilter(b_payload, (1,), zeroCrossing_3)
    #print(f"payload valid est len = {len(payload_valid_est) // map.N}")
    #Plot the OFDM packet
    plt.figure()
    plt.plot(file_iq, label = "OFDM Packet")
    plt.plot(M_values, label = "M Values")
    plt.plot(M_filter, label = "Filtered M")
    plt.plot(D, label = "Derivative of M_filter")
    #plt.plot(zeroCrossing_3, label = "zero crossings")
    #plt.plot(ignore_times, label = "Ignore window")
    plt.plot(preamble_valid_est, label = "Estiamtion of valid Sync")
    plt.plot(payload_valid_est, label = "Estimation of valid payload")
    plt.legend()
    plt.show()

    #get payload indicies
    payload_idx = np.where(payload_valid_est > 0)[0]
    sync_idx = np.where(preamble_valid_est > 0)[0]


    sync_ofdm_symbol = []
    for idx in sync_idx:
        sync_ofdm_symbol.append(file_iq[idx])
    print(f"sync len: {len(sync_ofdm_symbol)}")

    rx_ofdm_symbols = []
    for idx in payload_idx:
        rx_ofdm_symbols.append(file_iq[idx])
    sync_ofdm_symbol_np = np.array(sync_ofdm_symbol[:-1], dtype=complex)
    rx_ofdm_symbols_np = np.array(rx_ofdm_symbols, dtype=complex)



    print(len(rx_ofdm_symbols_np))
    chunks = np.split(rx_ofdm_symbols_np, N_data_symbols)
    #print(chunks[0])
    print(f"len of check is {len(chunks[0])}")
    sync_fft = np.fft.fft(sync_ofdm_symbol_np[:-4], map.N)
    plt.figure()
    plt.plot(np.abs(sync_fft))
    plt.title("sync FFT")

    qam_16 = ["0000",
              "0001",
              "0010",
              "0011",
              "0100",
              "0101",
              "0110",
              "0111",
              "1000",
              "1001",
              "1010",
              "1011",
              "1100",
              "1101",
              "1110",
              "1111"]
    
    qam_16_iq = []
    for word in qam_16:
        qam_16_iq.append(om.binary_to_iq(word))

    print(len(qam_16))  

    symbol1 = chunks[0]
    symbol1_fft = np.fft.fft(symbol1, map.N)
    plt.figure()
    plt.plot(np.real(symbol1_fft) , np.imag(symbol1_fft), '.', label = "Recieved OFDM packet")
    plt.plot(np.real(qam_16_iq) * np.sqrt(10), np.imag(qam_16_iq) * np.sqrt(10), '.', label = "Constalation Map")
    plt.show()


def generate_noise_buffer(buf_len:int = 200) -> np.ndarray:
    noise_buffer = []
    for i in range(buf_len):
        noise_buffer.append(random.uniform(-0.1,0.1))
    return np.array(noise_buffer, dtype=complex)


def normalize_samples(samples):
    max_magnitude = np.max(np.abs(samples))
    if max_magnitude > 0:
        return samples / max_magnitude
    else:
        return samples

if __name__ == "__main__":
    main()

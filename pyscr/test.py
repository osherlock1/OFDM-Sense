from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
import numpy as np
from sync_symbol import SyncSymbol
import matplotlib.pyplot as plt
import random


def generate_random_binary(N: int) -> list:
    
    binary_samples = ""
    for i in range(N):
        binary_samples += str(random.choice([0, 1]))
    return binary_samples

def parse_string(input_string, parse_length):
    chunks = []

    for i in range(0, len(input_string), parse_length):
        chunk = input_string[i:i + parse_length]
        chunks.append(chunk)
    return chunks

map = SubcarrierMap()
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


om = OFDMManager(SubcarrierMap())

data_symbol = OFDMSymbol(iq_samples48=iq_samples, pilots4=pilots, submap=map)
tx_data_block = om.create_tx_block(data_symbol)
sync = SyncSymbol()
sync_symbol = SyncSymbol()
tx_block = om.create_tx_block(sync)

send_signal = np.concatenate([tx_data_block, tx_block, tx_data_block, tx_data_block, tx_data_block])
P_values = []
M_values = []
R_values = []
for i in range(250):
    P, R, M = om.schmidl_cox_metrics_P_R_M(r = send_signal, delay = i)
    P_values.append(P)
    R_values.append(R)
    M_values.append(M)





input = generate_random_binary(48)
parsed_input = parse_string(input, 4)

iq_input = []

for word in parsed_input:
    iq_input.append(om.binary_to_iq(word))

print(iq_input)



def generate_random_packet():
    ofdm_data_symbols= [] 
    for i in range(5):

        input = generate_random_binary(48 * 4)
        parsed_input = parse_string(input, 4)

        iq_input = []
        for word in parsed_input:
            iq_input.append(om.binary_to_iq(word))

        input_array = np.array(iq_input, dtype = complex)
        ofdm_symbol = OFDMSymbol(iq_samples48=input_array, pilots4=pilots, submap=map)
        ofdm_symbol_time = om.create_tx_block(ofdm_symbol)
        ofdm_data_symbols.append(ofdm_symbol_time)

    final_packet = om.create_tx_block(SyncSymbol())

    for symbol in ofdm_data_symbols:
        final_packet = np.concatenate([final_packet, symbol])
    return final_packet


ofdm_packet = generate_random_packet()

time_finder = []
for n in range(48*12):
    P, R, M = om.schmidl_cox_metrics_P_R_M(ofdm_packet, n)
    time_finder.append(M)

plt.figure()
plt.plot(time_finder)
plt.plot(np.abs(ofdm_packet))
#plt.show()

"""
Store Array to File
"""
buffer_len = 200
buffer_noise = []
for i in range(buffer_len):
    buffer_noise.append(random.uniform(-1.0,1.0)
                        )
buffer = np.array(buffer_noise, dtype=complex)
ofdm_packet2 = np.concatenate([buffer,ofdm_packet,buffer])

# Save as interleaved I/Q float32 (IQIQIQ...)
iq_interleaved = np.empty(len(ofdm_packet2) * 2, dtype=np.float32)
iq_interleaved[0::2] = np.real(ofdm_packet2)  # I samples
iq_interleaved[1::2] = np.imag(ofdm_packet2)  # Q samples
iq_interleaved.tofile('data_files/ofdm_iq_interleaved.dat')

#assert ofdm_packet2.dtype == np.complex64

iq_cp64 = np.empty(ofdm_packet2.size * 2, dtype=np.complex64)
iq_cp64.tofile('data_files/ofdm_cp64.dat')
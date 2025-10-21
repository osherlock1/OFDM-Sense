import random
from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
import numpy as np
from sync_symbol import SyncSymbol


class DataGenerator:
    def __init__(self):
        map = SubcarrierMap()
        self.om = OFDMManager(map)
        self.pilots = np.array([1 + 0j, 1 + 0j, 1+0j, 1+0j], dtype=complex)

    def generate_random_binary(N: int) -> list:
        
        binary_samples = ""
        for i in range(N):
            binary_samples += str(random.choice([0, 1]))
        return binary_samples
    
    def _parse_string(input_string, parse_length):
        chunks = []

        for i in range(0, len(input_string), parse_length):
            chunk = input_string[i:i + parse_length]
            chunks.append(chunk)
        return chunks
    
    def generate_random_packet(self):
        ofdm_data_symbols= [] 
        for i in range(5):

            input = self.generate_random_binary(48 * 4)
            parsed_input = self.parse_string(input, 4)

            iq_input = []
            for word in parsed_input:
                iq_input.append(self.om.binary_to_iq(word))

            input_array = np.array(iq_input, dtype = complex)
            ofdm_symbol = OFDMSymbol(iq_samples48=input_array, pilots4=self.pilots, submap=map)
            ofdm_symbol_time = self.om.create_tx_block(ofdm_symbol)
            ofdm_data_symbols.append(ofdm_symbol_time)

        final_packet = self.om.create_tx_block(SyncSymbol())

        for symbol in ofdm_data_symbols:
            final_packet = np.concat([final_packet, symbol])
            
        return final_packet


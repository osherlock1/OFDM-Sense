import random
from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
import numpy as np
from sync_symbol import SyncSymbol


class DataGenerator:
    def __init__(self, seed = None):
        #Initiate helpers
        self.map = SubcarrierMap()
        self.om = OFDMManager()
        
        

        #Generate pilots array
        self.pilots_value = self.map.pilots_val
        self.pilots = np.ones(self.map.num_pilots, dtype=np.complex128) * self.pilots_value
        self.seed = seed
        if self.seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        

    def generate_random_binary(self, N: int) -> list:
        
        binary_samples = ""
        for i in range(N):
            binary_samples += str(random.choice([0, 1]))
        return binary_samples
    
    def _parse_string(self, input_string, parse_length):
        chunks = []

        for i in range(0, len(input_string), parse_length):
            chunk = input_string[i:i + parse_length]
            chunks.append(chunk)
        return chunks
    
    def generate_random_packet(self, N_data_symbols:int = 5) -> np.ndarray:
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        ofdm_data_symbols= [] 
        for i in range(N_data_symbols):
            size = len(self.map.data_bins) * 4
            input = self.generate_random_binary(size)
            parsed_input = self._parse_string(input, 4)

            iq_input = []
            for word in parsed_input:
                iq_input.append(self.om.binary_to_iq(word))

            input_array = np.array(iq_input, dtype = np.complex128)
            ofdm_symbol = OFDMSymbol(iq_samples=input_array, pilots_data=self.pilots)
            ofdm_symbol_time = self.om.create_tx_block(ofdm_symbol)
            ofdm_data_symbols.append(ofdm_symbol_time)

        final_packet = self.om.create_tx_block(SyncSymbol())

        for symbol in ofdm_data_symbols:
            final_packet = np.concatenate([final_packet, symbol])
            
        return final_packet

    def add_noise(self, x, snr_db, rng = None):
        if rng is None:
            rng = np.random.default_rng()
        Px = np.mean(np.abs(x) ** 2)
        snr_lin = 10 ** (snr_db / 10)
        sigma2 = Px / snr_lin
        n = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) * np.sqrt(sigma2/2)
        return x + n

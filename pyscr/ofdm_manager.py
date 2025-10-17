import numpy as np
from subcarrier_map import SubcarrierMap


class OFDMManager():
    def __init__(self, map:SubcarrierMap):
        self.map = map

    def ifft(self, symbol_freq:np.ndarray):
        """
        Compute IFFT of the QAM Frequency Packet (Default 64 Point)
        """
        symbol_freq.symbol = np.fft.ifft(symbol_freq.symbol, self.map.N)
        
    
    def fft(self, symbol_time:np.ndarray):
        """
        Compute FFT of time respose
        """
        symbol_time.symbol = np.fft.fft(symbol_time.symbol, self.map.N)
        
    
    def add_cycle_prefix(self, symbol:np.ndarray, prefix_len:int = 8) -> np.ndarray:
        """Add the cyclacle prefix the synbol (Default is 8)
        
            IMPORTANT! After you call this method the output will no longer be an OFDM symbol object for now I will just have
            it output a np array which will be the TX Block sent to the USRP
        
        """
        prefix = symbol.symbol[self.map.N-prefix_len:]
        TX_block = np.concatenate([prefix,symbol.symbol]).astype(complex)
        return TX_block
    
    def create_tx_block(self, symbol:np.ndarray)->np.ndarray:
        #Compute ifft
        self.ifft(symbol)
        #Add cycle prefix
        tx_block = self.add_cycle_prefix(symbol)
        return tx_block
    
    def binary_to_iq(self, binary: str, M: int = 16, scale_factor = np.sqrt(10)):
        k = np.log2(M) # Number of bits

        #Check if binary is 4 bits
        if len(binary) != k:
            raise ValueError(
                f"Binary Length Expected to be {k}"
                f" but got {len(binary)} instead"
            )

        grey_coded_map = {
            "00" : -3,
            "01" : -1,
            "11" : 1,
            "10" : 3
        }

        I = binary[0:2]
        Q = binary[2:4]
        #Build the IQ sample
        iq_sample = grey_coded_map[I] + 1j * grey_coded_map[Q]
        return iq_sample / scale_factor

    def iq_to_binary(self, iq_sample: complex, scale_factor = np.sqrt(10)):
        
        grey_coded_map = {
            -3 : "00",
            -1 : "01",
            1 : "11",
            3 : "10"
        }
        I = round(np.real(iq_sample) * scale_factor)
        Q = round((np.imag(iq_sample) * scale_factor))

        return grey_coded_map[I] + grey_coded_map[Q]
        
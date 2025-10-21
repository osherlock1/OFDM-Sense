import numpy as np
from subcarrier_map import SubcarrierMap


class OFDMManager():
    def __init__(self, map:SubcarrierMap):
        self.map = map
        self.N = map.N # Number of subcarries in a OFDM symbol

    def ifft(self, symbol_freq:np.ndarray):
        """
        Compute IFFT of the QAM Frequency Packet (Default 64 Point)
        """
        symbol_freq.symbol = np.fft.ifft(symbol_freq.symbol, self.N) * np.sqrt(self.N / (len(self.map.data_bins) + len(self.map.pilots_k)))
        
    
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
        """
        Create convert OFDM Symbol from Frequency Domain to Time Domain and add Cyclacle Prefix
        """
        #Compute ifft
        self.ifft(symbol)
        #Add cycle prefix
        tx_block = self.add_cycle_prefix(symbol)
        return tx_block
    
    def binary_to_iq(self, binary: str, M: int = 16, scale_factor = np.sqrt(10)):
        """
        Convert from 4 bit binary to 16-QAM gey-coded
        """
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
        """
        Convert an IQ sample to 16-QAM gey-coded
        """
        grey_coded_map = {
            -3 : "00",
            -1 : "01",
            1 : "11",
            3 : "10"
        }
        I = round(np.real(iq_sample) * scale_factor)
        Q = round((np.imag(iq_sample) * scale_factor))

        return grey_coded_map[I] + grey_coded_map[Q]
        
    def schmidl_cox_metrics(self, r: np.ndarray, delay: int):
        """
        Compute P, R, and M for the reciever Schmidl Cox Algorithm
        r = recieved time series data
        delay = shifting starting index
        """
        
        L = self.N // 2 #Half the number of subcarriers of OFDM symbols
        a = r[delay : delay + L]
        b = r[delay + L: delay + 2 * L]

        #Check for index bound errors
        if len(a) != L or len(b) != L:
            return 0j, 0.0, 0.0
        
        P = np.vdot(a , b)
        R = np.vdot(b, b).real
        M = (np.abs(P) ** 2) / (R ** 2 + 1e-12) #1e-12 to prevent division by 0
        
        return P, R, M



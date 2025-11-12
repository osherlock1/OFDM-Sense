import numpy as np
from subcarrier_map import SubcarrierMap


class OFDMManager():
    def __init__(self):
        self.map = SubcarrierMap()
        self.N = self.map.N # Number of subcarries in a OFDM symbol
        self.N_data_symbols = 5 #Number of payload symbols per OFDM packet


    def ifft(self, symbol_freq:np.ndarray):
        """
        Compute IFFT of the QAM Frequency Packet (Default 64 Point)
        """
        symbol_freq.symbol = np.fft.ifft(symbol_freq.symbol, self.N) #* np.sqrt(self.N) #/ (len(self.map.data_bins) + len(self.map.pilots_k)))
        
    
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
        Convert from 4 bit binary string to 16-QAM gey-coded
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

    def iq_to_binary(self, iq_sample: complex):
        """
        Convert an IQ sample to 16-QAM gey-coded
        """
        grey_coded_map = {
            -3 : "00",
            -1 : "01",
            1 : "11",
            3 : "10"
        }
        
        I = np.real(iq_sample)
        Q = np.imag(iq_sample)
        I_min = float('inf')
        Q_min = float('inf')
        I_map = 0
        Q_map = 0

        for key in grey_coded_map:
            I_delta = (np.abs(I - key))
            Q_delta = (np.abs(Q - key))


            if I_delta < I_min:
                I_map = key
                I_min = I_delta

            if Q_delta < Q_min:
                Q_map = key
                Q_min = Q_delta
        return grey_coded_map[I_map] + grey_coded_map[Q_map]





        #return grey_coded_map[I] + grey_coded_map[Q]
        
    def schmidl_cox_metrics_P_R_M(self, r: np.ndarray, delay: int):
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
        Ra = np.vdot(a, a).real
        Rb = np.vdot(b, b).real
        M = (np.abs(P) ** 2) / (Ra * Rb + 1e-12) #1e-12 to prevent division by 0
        
        return P, (Ra, Rb), M
    
    def build_ofdm_packet(self, iq_samples:np.ndarray, N_data_symbols:int = 5) -> np.ndarray:
        """
        TODO: FINISH THIS METHOD
        """
        ofdm_data_symbols = np.split(iq_samples, N_data_symbols)
        print(len(ofdm_data_symbols))


    def calc_BER(self, Y_ref, Y):
        """
        Calculate the Bit Error Rate
        Y_ref: Reference array of individual bits
        Y: Recieved array of unpacked bits from OFDM transfer
        returns bit error rate int
        """
        total_bits = len(Y_ref)
        errors = 0
        for i in range(len(Y_ref)):
            if Y_ref[i] != Y[i]:
                errors += 1
        
        #print(f"Number of Bit errors: {errors}")
        ber = errors/ total_bits
        return ber
    
    def calc_SER(self, Y_ref, Y):
        """
        Calculate Symbol Error Rate
        Y_ref: Reference array of IQ samples
        Y: Recieved array of IQ samples from transfer
        """

        Y_mapped = []
        for iq_sample in Y:
            iq_mapped = (self.calc_closesest_qam(iq_sample))
            Y_mapped.append(iq_mapped)
            
        Y_mapped = np.array(Y_mapped)


        errors = 0
        for i in range(len(Y)):
            if Y_ref[i] != Y_mapped[i]:
                errors += 1
       #print(f"ERRORS: {errors}")
        total_iq_samples = len(Y_ref)
        ser = errors / total_iq_samples
        return ser

    def calc_EVM(self, Y, Y_ref):
        """
        Calculate the Average Vector Magnitude Error
        """
        normalization = np.abs(np.sum(Y_ref))
        
        #Calc mean square error
        N = len(Y)
        #Mean refernce power
        sum_result = 0 
        for i in range(N):
            Ierr = np.real(Y[i]) - np.real(Y_ref[i])
            Qerr = np.imag(Y[i]) - np.imag(Y_ref[i])
            sum_result += ((Ierr ** 2) + (Qerr ** 2)) / (np.abs(Y_ref[i]) ** 2)
            #print(sum_result)
        evm = np.sqrt((1 / N) * sum_result)



        #Calc EVM
        #evm = np.sqrt(mean_square_error / mean_reference_power)
        return 20 * np.log10(evm)

    def decode_rx(self, Y) -> np.ndarray:
        """
        Method to Map and entire raw recieved OFDM iq samples and returns entire nearest mapping array
        Y = 
        """
        decoded_Y = []
        for iq_sample in Y:
            decded_sample = self.calc_closesest_qam(iq_sample)
            decoded_Y.append(decded_sample)
        return np.array(decoded_Y, dtype=complex)


    def calc_closesest_qam(self, iq_sample):
        """
        Method to map a single RX OFDM IQ sampel to its nearest reference map point.  Returns the nearest refence point
        iq_sample: single complex iq sample (SHOULD BE SCALED to QAM MAP i.e. [-3, -1, 1, 3])
        """
        I = np.real(iq_sample)
        Q = np.imag(iq_sample)

        qam_map = [-3 + 3j, -1 + 3j, 1 + 3j, 3 + 3j,
                   -3 + 1j, -1 + 1j, 1 +1j, 3 + 1j,
                   -3 - 1j, -1 - 1j, 1 - 1j, 3 - 1j,
                   -3 - 3j, -1 - 3j, 1 - 3j, 3 - 3j,]
        
        
        min_distance = float('inf')
        min_map_point = None
        for ref in qam_map:
            
            distance = np.abs(iq_sample - ref)
            if distance < min_distance:
                min_distance = distance
                min_map_point = ref
        return min_map_point

 

    def cfo_correct(self, Tx:np.ndarray, Rx:np.ndarray, fs:int, n_bins:int = 2 ** 12):
        
        #Define frequency bins
        f_grid = np.linspace(-fs/2, fs/2, n_bins, endpoint=False)

        #Define n array
        n_len = len(Tx)
        n = np.arange(n_len)

        G = np.empty(n_bins, dtype=np.complex128)
        #Calculate G array
        for i, f in enumerate(f_grid):
            r = Rx * np.exp(-1j * 2*np.pi * f * n / fs)
            G[i] = np.vdot(Tx,r) #Calculate correlation
        
        k = int(np.argmax(np.abs(G)))
        f_hat = f_grid[k]

        #Rx_hat = Rx * np.exp(-1j * 2*np.pi * f_hat * n / fs)


        return G, k, f_hat
    



        

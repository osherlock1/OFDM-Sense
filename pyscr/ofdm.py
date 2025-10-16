import numpy as np



class ofdmManager():
    def __init__(self):
        pass

    def ifft(self, samples, N:int = 64):
        """
        Compute IFFT of the QAM Frequency Packet (Default 64 Point)
        """
        response = np.fft.ifft(samples, N)
        return response
    
    
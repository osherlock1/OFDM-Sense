import numpy as np
from subcarrier_map import SubcarrierMap

class PilotSymbol():
    def __init__(self, seed:int = 10):

        self.map = SubcarrierMap()
        self.N = self.map.N #Expect N = 64
        self.symbol = np.zeros(SubcarrierMap().N, dtype = complex) #Subcarrier bins
        
        self.used_k = self.map.data_bins

        self.seed = seed
        #Build deterministic BPSK sequence
        np.random.seed(self.seed)
        self.bpsk_seq = self._BPSK_form()

        #Fill in even bins
        self._add_even(self.used_k)
        
 

    def _BPSK_form(self):  
        """
        Generated random binary numbers to fill in the even indicies of the sync symbol
        """      
        np.random.seed(self.seed)
        bpsk_seq = np.random.choice([-1, 1], size=int(self.N))
        return(bpsk_seq)
        
    def _add_even(self, used_k):
        """
        Add random BPSK values to even indicies of the sync symbol
        """
        i = 0
        for k in used_k:
            self.symbol[self.map.idx(k)] = self.bpsk_seq[i]
            i += 1


    
    
import numpy as np
from subcarrier_map import SubcarrierMap

class SyncSymbol():
    def __init__(self):

        map = SubcarrierMap()
        self.N = map.N #Expect N = 64
        self.symbol = np.zeros(SubcarrierMap().N, dtype = complex) #Subcarrier bins
        
        #Define Used frequency bins
        used_neg = list(range(-26, 0))
        used_pos = list(range(1, 27))
        used_k = used_neg + used_pos

        #Get Even Indicies
        even_k = [k for k in used_k if (k % 2 == 0)]
        
        #Build deterministic BPSK sequence
        np.random.seed(42)
        self.bpsk_seq = self._BPSK_form()

        #Fill in even bins
        self._add_even(even_k)
        
 

    def _BPSK_form(self):  
        """
        Generated random binary numbers to fill in the even indicies of the sync symbol
        """      
        bpsk_seq = np.random.choice([-1, 1], size=int(self.N // 2))
        return(bpsk_seq)
        
    def _add_even(self, even_k):
        """
        Add random BPSK values to even indicies of the sync symbol
        """
        i = 0
        for k in even_k:
            self.symbol[SubcarrierMap().idx(k)] = self.bpsk_seq[i]


    
    

    
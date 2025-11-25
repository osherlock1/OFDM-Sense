import numpy as np
from subcarrier_map import SubcarrierMap

class OFDMSymbol:
    def __init__(self, iq_samples: np.ndarray, pilots_data: np.ndarray):
        
        self.submap = SubcarrierMap()
        
        self.iq_samples = iq_samples
        self.pilots_k = self.submap.pilots_k
        self.pilots_idx = np.array([self.submap.idx(k) for k in self.pilots_k])
        self.data_k = self.submap.data_bins
        self.pilots_data = pilots_data 
        self.symbol = np.zeros(self.submap.N, dtype=complex) #initiate the ofdm array
        #Check if data samples matches subcarrier map
        if len(self.iq_samples) != len(self.data_k):
            raise ValueError(
                f"Expected {len(self.data_k)} iq samples"
                f"but got {len(self.iq_samples)} instead"
            )

        #Check if pilot values amout matches expected from subcarrier map
        if len(self.pilots_data) != len(self.pilots_k):
            raise ValueError(
                f"Expected {len(self.pilots_k)} pilot values"
                f"but got {len(self.pilots_data)} instead"
            )
        self._build_ofdm_symbol(self.iq_samples, self.pilots_data)
        #print("OFDM Symbol Instantiated!")

    # HELPER METHODS
    def _build_ofdm_symbol(self, data, pilots):
        """
        Build the completed OFDM Symbol
        """
        self._add_data(iq_samlpes=data)
        self._add_pilots(pilots)

    def _add_data(self, iq_samlpes):
        """
        Add Data Samples to X(OFDM Symbol)
        """
        i = 0
        for k in self.data_k:
            self.symbol[self._idx(k)] = iq_samlpes[i]
            i += 1

    def _add_pilots(self, pilots):
        """
        Add pilots to proper indicies
        """
        i = 0
        for k in self.pilots_k:
            self.symbol[self._idx(k)] = pilots[i]
            i += 1

    def _idx(self, idx:int) -> int:
        return idx % self.submap.N
    
    @property
    def pilots(self) -> np.ndarray:
        return self.symbol[np.array(self.pilots_idx)]
    
    @property
    def data(self) -> np.ndarray:
        return self.symbol[np.array(self.data_k)] #FIXME: THIS IS WRONG 

        
    
    


    
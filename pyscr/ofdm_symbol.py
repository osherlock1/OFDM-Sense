import numpy as np
from subcarrier_map import SubcarrierMap

class OFDMSymbol:
    def __init__(self, iq_samples48: np.ndarray, pilots4: np.ndarray):
        self.iq_samlpes48 = iq_samples48
        self.pilots4 = pilots4
        self.submap = SubcarrierMap()

        self.symbol = np.zeros(64, dtype=complex) #initiate the ofdm array

        #Check if data samples matches subcarrier map
        if len(self.iq_samlpes48) != len(self.submap.data_bins):
            raise ValueError(
                f"Expected {len(self.submap.data_bins)} data sybols"
                f"but got {len(self.iq_samlpes48)} instead"
            )

        #Check if pilot values amout matches expected from subcarrier map
        if len(self.pilots4) != len(self.submap.pilots_k):
            raise ValueError(
                f"Expected {len(self.submap.pilots_k)} pilot values"
                f"but got {len(self.pilots4)} instead"
            )
        self._build_ofdm_symbol(self.iq_samlpes48, self.pilots4)
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
        for k in self.submap.data_bins:
            self.symbol[self._idx(k)] = iq_samlpes[i]
            i += 1

    def _add_pilots(self, pilots):
        """
        Add pilots to proper indicies
        """
        i = 0
        for k in self.submap.pilots_k:
            self.symbol[self._idx(k)] = pilots[i]
            i += 1

    def _idx(self, idx:int) -> int:
        return idx % self.submap.N

        
    
    


    
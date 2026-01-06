from dataclasses import dataclass
import random


@dataclass
class OFDMConfig:
    N: int = 128
    CP_LEN: int = 16
    GUARD_LEN: int = 6 #This should be even
    FS: float = 100e6
    #Defined below
    data_carriers: list = None 
    pilot_carriers: list = None

    def __post_init__(self):
        # logical initialization
        if self.data_carriers is None or self.pilot_carriers is None:
            self._load_random_map()

        #VALIDATION
        overlap = set(self.data_carriers).intersection(set(self.pilot_carriers))
        if overlap:
            raise ValueError(f"Configuration Error: Indicies {overlap} are defined as both pilot and data.")

    def _load_random_map(self):
        """
        Generate semi-random pilot and data symbol indexes
        """
        #Set random seed
        random.seed(42)

        #Define Active Bins
        used_neg = list(range(-(self.N // 2) + self.GUARD_LEN, 0))
        used_pos = list(range(1,(self.N // 2) - self.GUARD_LEN))
        active_bins = used_neg + used_pos

        #Select Pilots
        num_pilots = self.N // 4 #Ratio of total subcarriers to data subcarriers
        pilots_k = random.sample(active_bins, num_pilots)

        #Select Data
        data_k = [k for k in active_bins if k not in pilots_k]

        #Convert k to idx
        self.pilot_carriers = [self._idx(k) for k in pilots_k]
        self.data_carriers = [self._idx(k) for k in data_k]

        #Sort for plotting
        self.pilot_carriers.sort()
        self.data_carriers.sort()



    def _load_default_map(self):
        """
        Define Default OFDM mapping if no Config is provided
        """
        #Standard OFDM mapping
        # - 48 Data Subcarriers
        # - 4 Pilot Subarriers
        # - 1 DC (Index 0)
        # - 11 Guard Carriers

        #Define PIlots
        pilot_indicies = [-21, -7, 7, 21]
        self.pilot_carriers = [self]
        pass
    
    def _idx(self, k: int) -> int:
        """
        Helper to convert from python indexing to freq bin indexing
        """
        return (k + self.N) % self.N

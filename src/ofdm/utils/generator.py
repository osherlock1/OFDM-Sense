import random
import numpy as np
from dataclasses import dataclass
import random
from typing import Tuple, List
from ofdm.config import OFDMConfig


class DataGenerator:
    def __init__(self, config: OFDMConfig, seed:int = None):
        self.config = config
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
    
    def generate_random_payload(self) -> Tuple[np.ndarray, str]:
        """
        Generates random QAM16 data from random binary
        return the iq data and binary string as a tuple for verification
        """

        #Calculate necissary number of bits
        n_data_bins = len(self.config.data_carriers)
        bits_needed = n_data_bins * 4 #Each QAM iq sample is 4 bits

        #Generate Bits
        bits = "".join([str(random.choice([0, 1])) for i in range(0, len(bits), 4)])

        #Map to Qam
        #FIXME: NEED TO ADD QAM MAPPING TO MODULATION
import random
import numpy as np
from typing import Tuple
from ofdm.config import OFDMConfig
from ofdm.modulation import qam


class DataGenerator:
    def __init__(self, config: OFDMConfig, seed: int = None):
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

        # Calculate necissary number of bits
        n_data_bins = len(self.config.data_carriers)
        bits_needed = n_data_bins * 4  # Each QAM iq sample is 4 bits

        # Generate Bits
        bits = "".join([str(random.choice(["0", "1"])) for _ in range(bits_needed)])

        # Parse Binary string into 4 bit words
        chunks = [bits[i : i + 4] for i in range(0, len(bits), 4)]

        # Convert Bianry 4 bit words to iq samples
        iq_data = np.array([qam.binary_to_iq(chunk) for chunk in chunks])

        return iq_data, bits

import random
import numpy as np

class SubcarrierMap:
    N: int = 64
    num_pilots = 12
    random.seed(42)


    @property
    def pilots_k(self):
        random.seed(42)
        used_neg = list(range(-26, 0))
        used_pos = list(range(1,27))
        pilot_bins = []
        used_p_bins = used_neg + used_pos

        for k in range(self.num_pilots):
            random_k = random.choice(used_p_bins)
            pilot_bins.append(random_k)
        return pilot_bins


    cp_len = 8
    def idx(self, k:int) -> int:
        return k % self.N
    
    @property
    def data_bins(self):
        used_neg = list(range(-26, 0))
        used_pos = list(range(1,27))
        used_bins = []
        
        for k in (used_neg + used_pos):
            if k not in self.pilots_k:
                used_bins.append(k)
        return used_bins

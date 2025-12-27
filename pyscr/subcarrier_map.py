import random
import numpy as np

class SubcarrierMap:

    #-------------------
    # OFDM SYMBOL MAPPING
    #--------------------
    N = 2 ** 10
    num_pilots = N // 4
    cp_len = 20
    guard_len = 5



    random.seed(42)


    @property
    def pilot_vals(self):
        random.seed(42)
        qam16_vals = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        pilot_vals = []
        for i in range(self.num_pilots):
            pilot_vals.append(random.choice(qam16_vals))
        return pilot_vals

        
    #FIXME: HARDCODED USED NEG AND USED POS VALUES

    @property
    def pilots_k(self):
        random.seed(42)
        used_neg = list(range(-(self.N // 2) + self.guard_len, 0))
        used_pos = list(range(1,(self.N // 2) - self.guard_len))
        pilot_bins = []
        used_p_bins = used_neg + used_pos

        return random.sample(used_p_bins, self.num_pilots)


    
    def idx(self, k:int) -> int:
        return k % self.N
    
    @property
    def data_bins(self):
        used_neg = list(range(-(self.N // 2) + self.guard_len, 0))
        used_pos = list(range(1,(self.N // 2) - self.guard_len))
        used_bins = []
        
        for k in (used_neg + used_pos):
            if k not in self.pilots_k:
                used_bins.append(k)
        return used_bins

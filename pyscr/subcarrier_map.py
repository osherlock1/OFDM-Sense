class SubcarrierMap:
    N: int = 64
    pilots_k = (-21, -7, 7, 21)
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

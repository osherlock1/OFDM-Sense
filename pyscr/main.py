from ofdm import ofdmManager
import matplotlib.pyplot as plt
import numpy as np
from subcarrier_map import SubcarrierMap
from ofdm_symbol import OFDMSymbol
#Indicies helper
def idx(k):
    return k % 64



def main():


    map = SubcarrierMap()

     #Simple IQ samples respoinse
    iq_samples = np.array([-0.9487 + 0.9487j, -0.3162 + 0.9487j,  0.3162 + 0.9487j,  0.9487 + 0.9487j,
 -0.9487 + 0.3162j, -0.3162 + 0.3162j,  0.3162 + 0.3162j,  0.9487 + 0.3162j,
 -0.9487 - 0.3162j, -0.3162 - 0.3162j,  0.3162 - 0.3162j,  0.9487 - 0.3162j,
 -0.9487 - 0.9487j, -0.3162 - 0.9487j,  0.3162 - 0.9487j,  0.9487 - 0.9487j,

 -0.9487 + 0.9487j, -0.3162 + 0.9487j,  0.3162 + 0.9487j,  0.9487 + 0.9487j,
 -0.9487 + 0.3162j, -0.3162 + 0.3162j,  0.3162 + 0.3162j,  0.9487 + 0.3162j,
 -0.9487 - 0.3162j, -0.3162 - 0.3162j,  0.3162 - 0.3162j,  0.9487 - 0.3162j,
 -0.9487 - 0.9487j, -0.3162 - 0.9487j,  0.3162 - 0.9487j,  0.9487 - 0.9487j,

 -0.9487 + 0.9487j, -0.3162 + 0.9487j,  0.3162 + 0.9487j,  0.9487 + 0.9487j,
 -0.9487 + 0.3162j, -0.3162 + 0.3162j,  0.3162 + 0.3162j,  0.9487 + 0.3162j,
 -0.9487 - 0.3162j, -0.3162 - 0.3162j,  0.3162 - 0.3162j,  0.9487 - 0.3162j,
 -0.9487 - 0.9487j, -0.3162 - 0.9487j,  0.3162 - 0.9487j,  0.9487 - 0.9487j], dtype=complex)


    pilots = np.array([1 + 0j, 1 + 0j, 1+0j, 1+0j], dtype=complex)

    ofdm_symbol1 = OFDMSymbol(iq_samples48=iq_samples, pilots4=pilots, submap=map)
    
    print(ofdm_symbol1.symbol)
    om = ofdmManager()

    #Define the OFDM symbol
    X = np.zeros(64, dtype=complex)

    used_neg = list(range(-26,0))
    used_pos = list(range(1,27))
    pilots_k = [-21,-7,7,21]

    data_bins = []
    
    for k in (used_neg + used_pos):
        if k not in pilots_k:
            data_bins.append(k)


    #Fill in X with the data samples
    i = 0
    for k in data_bins:
        X[idx(k)] = iq_samples[i]
        i += 1

   
    



if __name__ == "__main__":
    main()



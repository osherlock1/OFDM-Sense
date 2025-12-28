import numpy as np
from ofdm.config import OFDMConfig

def generate_sync_symbol(config: OFDMConfig, seed: int = 42) -> np.ndarray:
    """
    Generates Shmidl-Cox Syncronization Symbol
    """
    np.random.seed(seed)

    #Initiate Emptry Grid
    symbol_freq = np.zeros(config.N, dtype=complex)

    #Define the Used freq bins
    used_neg = list(range(-(config.N // 2) + config.GUARD_LEN, 0))
    used_pos = list(range(1,(config.N // 2) - config.GUARD_LEN))
    used_k = np.array(used_neg + used_pos) #Combine and convert to ndarray
    
    #Get Even Indicies
    even_k = used_k[used_k % 2 == 0]
    
    #Build deterministic BPSK sequence
    n_size = len(even_k != 0)
    bpsk_seq = np.random.choice([-3 / np.sqrt(10), 3 / np.sqrt(10)], size = n_size)


    print(bpsk_seq)
    print(f"Lenth of BPSK is {len(bpsk_seq)}")
    print(f"Length of even_k is {len(even_k)}")
    #symbol_freq[even_k] = bpsk_seq

    symbol_freq[config._idx(even_k)] = bpsk_seq
    return symbol_freq
    


    pass

def generate_pilot_symbol(config: OFDMConfig, seed: int =42) -> np.ndarray:
    pass
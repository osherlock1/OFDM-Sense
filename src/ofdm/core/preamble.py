import numpy as np
from ofdm.config import OFDMConfig
import random

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

    symbol_freq[config._idx(even_k)] = bpsk_seq
    return symbol_freq
    

def generate_pilot_symbol(config: OFDMConfig, seed: int =42) -> np.ndarray:
    """
    Generated Pilot Symbol for Channel Estimation and CFO Calibration
    """
    np.random.seed(seed)

    #Initiate empty symbol
    symbol_freq = np.zeros(config.N, dtype=complex)

    #Defined Used Freq Bins
    #used_k = np.array(config.data_carriers)
    used_idx = np.arange(config.N)
    used_k = config._idx(used_idx)

    
    #Generate Random data
    n_size = len(used_k)
    bpsk_seq = np.random.choice([-3, -1, 1, 3], size = n_size)

    symbol_freq[config._idx(used_k)] = bpsk_seq / np.sqrt(10)
    return symbol_freq

def generate_pilot_vals(config: OFDMConfig, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    
    #Define Possile Values
    qam16_vals = np.array([-3, -1, 1, 3]) / np.sqrt(10)

    #Number of pilot values
    n_pilots = len(config.pilot_carriers)

    pilot_values = [random.choice(qam16_vals) for _ in range(n_pilots)]

    return pilot_values
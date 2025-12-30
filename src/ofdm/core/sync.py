import numpy as np
import scipy.signal
from typing import Tuple, Optional
from ofdm.config import OFDMConfig

#Temp debugg
from ofdm.viz import plotter
import matplotlib.pyplot as plt

def calculate_schmidl_cox_metrics(rx_signal: np.ndarray, config:OFDMConfig)->Tuple[np.ndarray, np.ndarray]:
    """  
    Calculate the M and P metrics packet dection and syncronization
    """

    L = config.N // 2

    #Correlate the signal with a version of itself shifted by L
    r_upper = rx_signal[L:]
    r_lower = rx_signal[:-L]

    #Multiply elemeent wise
    mult_term = np.conj(r_lower) * r_upper

    #Moving sum over window L
    P = scipy.signal.fftconvolve(mult_term, np.ones(L), mode='valid')

    #Calculate R
    energy = np.abs(r_upper) ** 2
    R = scipy.signal.fftconvolve(energy, np.ones(L), mode='valid')
    
    #Compute M Metric
    min_len = min(len(P), len(R))
    P = P[:min_len]
    R = R[:min_len]

    #Avoid division by zero
    R[R == 0] = 1e-10

    #Calculate M = |P|^2 / R^2 with energy threshhold
    M = np.zeros_like(R)

    #Calculate nosie floor (cut out anything below 10% of avg power)
    avg_energy = np.mean(R)
    energy_threshold = avg_energy * 0.1

    valid_indicies = R > energy_threshold



    M[valid_indicies] = (np.abs(P[valid_indicies]) ** 2) / (R[valid_indicies] ** 2)
    return M, P


def find_start_idx(M: np.ndarray, config: OFDMConfig, threshold: float = 0.8) -> int:
    """  
    Find the start of the packet based on the M metric
    """

    canidates = np.where(M > threshold)

    if len(canidates) == 0:
        #Pick max
        start_idx = np.argmax(M)
        return start_idx
    
    peak_idx = np.argmax(M)
    return peak_idx

def estimate_cfo_coarse(P_value: complex, config: OFDMConfig) -> float:
    #Calculate Phase difference with P metrics
    theta = np.angle(P_value)
    cfo_hz = (theta * config.FS) / (np.pi * config.N)
    return cfo_hz
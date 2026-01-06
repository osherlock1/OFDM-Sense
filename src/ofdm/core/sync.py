import numpy as np
import scipy.signal 
from typing import Tuple, Optional
from ofdm.config import OFDMConfig
from scipy.signal import correlate

#Temp debugg
from ofdm.viz import plotter
import matplotlib.pyplot as plt

def calculate_schmidl_cox_metrics(rx_signal: np.ndarray, config:OFDMConfig)->Tuple[np.ndarray, np.ndarray]:
    """  
    Calculates the Schmidl-Cox Syncronization algorithm M and P metrics packet dection and syncronization

    Args:s
        rx_signal: Recieved raw time series data
        config: OFDMConfig()

    Returns:
        M_metric: The timing metric
        P_metric: The cross correlation metric
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


def find_start_idx(
    M_metric: np.ndarray, 
    config: OFDMConfig, 
    rx_signal: np.ndarray,
    known_sync_time: np.ndarray,
    search_window: int = 20
    ) -> int:
    """  
    Determine the exact sample index where the packet begins (Start of CP).
    Args:
        rx_signal: The raw recieved complex samples
        M_metric: The Schmidl_Cox metric (get from calculate_schmidl_cox_metrics())
        known_sync_time: Ideal time-domain sync symbol (with CP)
        search_window: How many samlpes Left and Right to search during fine sync
    Returns:
        The exact idx of the packet start (beginning of CP)
    """

    #-------- Coarse Sync ---------
    coarse_symbol_start = np.argmax(M_metric)

    #Estimate packet start
    coarse_packet_start = coarse_symbol_start - config.CP_LEN
    
    #Prevent negative index
    if coarse_packet_start < 0:
        coarse_packet_start = 0

    #-------- Fine Sync ---------
    start_search = max(0, coarse_packet_start - search_window)
    end_search = min(len(rx_signal), coarse_packet_start + len(known_sync_time) + search_window)

    rx_chunk = rx_signal[start_search : end_search]

    #Cross-Correlate
    corr = correlate(rx_chunk, known_sync_time, mode='valid')

    fine_offset = np.argmax(np.abs(corr))

    #Calculate final index
    exact_start_index = start_search + fine_offset

    return exact_start_index


def estimate_cfo_coarse(P_value: complex, config: OFDMConfig) -> float:
    #Calculate Phase difference with P metrics
    theta = np.angle(P_value)
    cfo_hz = (theta * config.FS) / (np.pi * config.N)
    return cfo_hz
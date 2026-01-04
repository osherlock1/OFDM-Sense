import numpy as np



def matched_filter_calc(rx_iq:np.ndarray, ref_iq:np.ndarray, fs:float) -> np.ndarray:
    """  
    Calculated the delay through matched filter delay estimation
    
    """
    z = np.correlate(rx_iq, ref_iq, mode="full")

    lags = np.arange(-len(rx_iq) + 1, len(ref_iq)) / fs
    
    return z, lags
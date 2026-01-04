import numpy as np



def matched_filter_calc(rx_iq:np.ndarray, ref_iq:np.ndarray) -> np.ndarray:
    """  
    Calculated the delay through matched filter delay estimation
    
    """
    z = np.correlate(rx_iq, ref_iq, mode="full")
    return z
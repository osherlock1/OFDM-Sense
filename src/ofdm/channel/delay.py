import numpy as np


def matched_filter_calc(rx_iq:np.ndarray, ref_iq:np.ndarray, fs:float) -> np.ndarray:
    """  
    Calculate the delay through matched filter delay estimation

    Args:
        rx_iq: Recieved IQ data
        ref_iq: Reference IQ data
        fs: Sampling Frequency
    
    Returns:
        z: Complex corss-correlation output
        lags: Time lag corresponding to each element of z, in seconds
    """
    z = np.correlate(rx_iq, ref_iq, mode="full")
    lags = np.arange(-len(rx_iq) + 1, len(ref_iq)) / fs
    return z, lags

def upsample(raw_data:np.ndarray, scale_factor:int = 100)->np.ndarray:
    """
    Upsamples raw data based on the scale factor for sub sampling matched filter delay estimation

    Args:
        raw_data: Raw pilot symbol data to be upsampled
        scale_factor: Multiplication factor for umsampling i.e. to upsample from 100MHz to 10Ghz use 100
    
    Returns:
        Upsampled np.ndarray data
    """
    n_data = len(raw_data)
    n_padded = n_data * scale_factor

    freq = np.fft.fftshift(np.fft.fft(raw_data))
    n_total_zeros = n_padded - n_data
    zeros_one_side = np.zeros(n_total_zeros // 2)
    freq_zero_padded_shifted = np.fft.ifftshift(np.concatenate([zeros_one_side, freq, zeros_one_side]))
    return np.fft.ifft(freq_zero_padded_shifted) * scale_factor

def scale_rx_signal(raw_rx_data:np.ndarray)->np.ndarray:
    max_val = np.max(np.abs(raw_rx_data))
    if max_val > 0:
        return raw_rx_data * (0.9) / max_val
    return raw_rx_data
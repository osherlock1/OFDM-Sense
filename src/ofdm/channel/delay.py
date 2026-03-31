import numpy as np
import scipy
import scipy.signal

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

def zero_padded_upsample(raw_data:np.ndarray, scale_factor:int = 100)->np.ndarray:
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
    """
    Scales an array to +- 0.9
    """
    max_val = np.max(np.abs(raw_rx_data))
    if max_val > 0:
        return raw_rx_data * (0.9) / max_val
    return raw_rx_data

def _correlate(rx_signal, ref_signal):
    corr = scipy.signal.correlate(rx_signal, ref_signal, mode='full')
    lags = scipy.signal.correlation_lags(len(rx_signal), len(ref_signal), mode='full')
    peak_idx = np.argmax(np.abs(corr))
    return corr, lags, peak_idx

def calculate_sub_sample_delay_parabolic(rx_signal:np.ndarray, ref_signal:np.ndarray, fs:float)->float:
    """
    Calculates sub-sample delay using Parabolic Interpolation

    Correlates rx_signal against ref_signal, then upsamples using parabolic interpolation.

    Args:
        rx_signal: Recieved IQ data
        ref_signal: Known reference signal (e.g. TX pilot symbol)
        fs: Sampling Frequency in HZ

    Returns:
        Estimated delay in seconds
    """
    corr, lags, peak_idx = _correlate(rx_signal, ref_signal)
    corr_mag = np.abs(corr)

    if peak_idx > 0 and peak_idx < len(corr) - 1:
        alpha = corr_mag[peak_idx - 1]
        beta = corr_mag[peak_idx]
        gamma = corr_mag[peak_idx + 1]
        fractional_shift = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
    else:
        fractional_shift = 0.0

    precise_lag = lags[peak_idx] + fractional_shift
    return precise_lag / fs


def calculate_sub_sample_delay_zp(rx_signal:np.ndarray, ref_signal:np.ndarray, fs:float, upsample_factor:int = 100, search_radius:int = 16)->float:
    """
    Calculates sub sample delay using zero-padded FFT interpolation

    Correlates rx_signal against ref_signal, then upsamples a window around the correlation peak via zero-padding
    in the frequency domain to achieve sub-sample timing resolution

    Args:
        rx_signal: Recieved IQ samples
        ref_signal: Reference IQ samples (e.g. known TX pilot symbol)
        fs: Sampling frequency in HZ
        upsample_factor: Interpolation factor applied to the correlation window
        search_radius: Nuber of samples on each side of the coarse peak to search

    Returns:
        Estimated delay in seconds
    """
    corr, lags, peak_idx = _correlate(rx_signal, ref_signal)

    start = max(0, peak_idx - search_radius)
    end = min(len(corr), peak_idx + search_radius)
    search_window = corr[start:end]

    window_upsampled = zero_padded_upsample(search_window, scale_factor=upsample_factor)
    upsampled_peak_idx = np.argmax(window_upsampled)

    fractional_offset = upsampled_peak_idx / upsample_factor
    final_idx = start + fractional_offset

    zero_lag_index = np.where(lags == 0)[0][0]
    final_lag_samples = final_idx - zero_lag_index
    return final_lag_samples / fs


import numpy as np
from ofdm.config import OFDMConfig


def freq_to_time(freq_data:np.ndarray) -> np.ndarray:
    """  
    Convert OFDM Symbol from freq domain to time signal
    """
    return np.fft.ifft(freq_data)

def time_to_freq(time_signal:np.ndarray) -> np.ndarray:
    """  
    Convert time signal into frequency domain
    """
    return np.fft.fft(time_signal)

def add_cp(time_signal:np.ndarray, cp_len: int = OFDMConfig.CP_LEN) -> np.ndarray:
    """  
    Takes N samples from the end of the signal and prepends it to the start
    """

    if cp_len <= 0:
        return time_signal
    
    #Get the prefix from the end
    prefix = time_signal[-cp_len:]

    #Concatenate
    return np.concatenate([prefix, time_signal])

def remove_cp(time_signal_with_cp: np.ndarray, cp_len: int = OFDMConfig.CP_LEN) -> np.ndarray:
    """  
    Removes the CP from the start of the time signal
    """

    if cp_len <= 0:
        return time_signal_with_cp
    
    return time_signal_with_cp[cp_len:]

def create_time_domain_symbol(freq_data: np.ndarray, cp_len:int = OFDMConfig.CP_LEN) -> np.ndarray:
    """
    Helper to quickly go from a freq OFDM symbol and convert it into a time domain signal
    with the cp prepended ready to be packed into the full ofdm packet
    """
    #Convert to Time
    time_signal = freq_to_time(freq_data)

    #Add Cp
    final_time_symbol = add_cp(time_signal, cp_len)

    return final_time_symbol


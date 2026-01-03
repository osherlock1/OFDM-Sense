import numpy as np
from typing import Tuple
#Internal
from ofdm.config import OFDMConfig
from ofdm.core import preamble, payload
def estimate_cfo(
        tx_ref: np.ndarray,
        rx_signal: np.ndarray,
        fs: float,
        search_window:int = 10,
        n_bins: int = 4096
) -> Tuple[float, np.ndarray]:
    """  
    Calculates carrier frequency offest (CFO) 
    
    Args:
        tx_ref: reference signal (Time Domain)
        rx_signal recieved signal (Time Doamin)
        fs: sample frequency
        n_bins: FFT resolution
        search_window: +- how many time delays to search
    
    Returns: 
        best_cfo: Estiamted CFO
        best_delay: The sample offset relative to start of rx_window)
    """
    #Prepare arrays
    n_ref = len(tx_ref)

    #Store max peaks
    global_max_mag = -1.0
    best_cfo = 0.0
    best_delay = 0

    heatmap = []

    possible_delays = range(0, len(rx_signal) - n_ref)

    for delay_idx in possible_delays:
        #Slice RX signal based on delay
        rx_chunk = rx_signal[delay_idx: delay_idx + n_ref]

        #Compute Mixing Product
        mixing_product = tx_ref * np.conj(rx_chunk)

        #FFT to find dominant Frequency
        spectrum = np.fft.fft(mixing_product, n=n_bins)
        spectrum_shifted = np.fft.fftshift(spectrum)
        magnitude = np.abs(spectrum_shifted)

        #Save for plotting
        heatmap.append(magnitude)

        #Check for new record
        current_max = np.max(magnitude)
        if current_max > global_max_mag:
            global_max_mag = current_max
            best_delay = delay_idx

            #Find frequency of peak
            peak_f_idx = np.argmax(magnitude)
            freq_axis = np.fft.fftshift(np.fft.fftfreq(n_bins, d=1/fs))
            best_cfo = freq_axis[peak_f_idx]
    
    return best_cfo, best_delay, np.array(heatmap)


def apply_cfo(rx_signal:np.ndarray, cfo:float, fs:float) -> np.ndarray:
    """  
    Correct for CFO on recieved Signal

    Args:
        rx_signal: Recieved Symbol (No CP)
        cfo: Calculated Frequency offest from estimate_cfo()
        fs: Sampling Frequency
    
    Returns:
        Symbol corrected
    """
    t = np.arange(len(rx_signal)) / fs
    correction_vector = np.exp(-1j * 2 * np.pi * cfo * t)
    return rx_signal * correction_vector

def prepare_data_symbol(rx_signal:np.ndarray, config:OFDMConfig)->np.ndarray:
    """  
    Builds the Pilot only 

    Args:
        rx_singal: Recieved data symbol no cp (Time Domain)
        config: OFDMConfig()

    Returns:
        rxn_time: rx time signal with non pilot subcarriers zeroed out
        txn_time: tx ref time signal with non pilot subcarriers zeroed out
    """
    #Pilot and Data Idx
    pilot_idx = config._idx(np.array(config.pilot_carriers))
    data_idx = config._idx(np.array(config.data_carriers))

    #Create TX Ref
    pilot_vals = preamble.generate_pilot_vals(config=config)
    TXk_pilots = np.zeros_like(rx_signal, dtype=complex)
    TXk_pilots[pilot_idx] = pilot_vals
    
    #Convert RX Signal to Frequency Domain
    RXk = np.fft.fft(rx_signal)

    #Zero out non pilots
    RXk_pilots = np.zeros_like(RXk, dtype=complex)
    RXk_pilots[pilot_idx] = RXk[pilot_idx]

    #Convert to time domain
    txn_time = np.fft.ifft(TXk_pilots)
    rxn_time = np.fft.ifft(RXk_pilots)

    return rxn_time, txn_time

def estimate_phase_error(tx_ref:np.ndarray, rx_signal:np.ndarray):

    #Calculate correlation
    correlation = np.vdot(tx_ref, rx_signal)

    #Get Phase Error
    phase_error = np.angle(correlation)

    return phase_error
    
import numpy as np
from ofdm.config import OFDMConfig

def channel_estimation_calc(rx_pilot_freq:np.ndarray, tx_pilot_ref: np.ndarray, config:OFDMConfig) -> np.ndarray:
    """  
    Performs Channel Estimation and returns subcarrier channel gains
    
    Args:
        rx_pilot_freq: Recieved RX Symbol no CP (After CFO Correction)
        tx_pilot_ref: Reference Pilot Symbol from TX

    Returns:
        Lambda_est: Channel Gains
    """

    #Identify Bins carrying data
    active_bins = np.concatenate([config.data_carriers, config.pilot_carriers])
    active_bins = np.unique(active_bins) #Sort

    #Initialize lambda
    Lambda_est = np.ones_like(rx_pilot_freq, dtype=complex)

    #Estiamtion channels
    Lambda_est[active_bins] = rx_pilot_freq[active_bins] / (tx_pilot_ref[active_bins] + 1e-12)

    return Lambda_est

def apply_gains(rx_symb_freq:np.ndarray, Lambda_est:np.ndarray)-> np.ndarray:
    """  
    Apply Channel gains to RX signal
    
    Args:
        rx_symb_freq: Recieved RX symbol after CFO with no CP
        Lamba_est: Channel gains calculated from channel_estimation_calc()
    """
    safe_lambda = Lambda_est.copy()
    threshold = 1e-2
    safe_lambda[np.abs(safe_lambda) < threshold] = 1.0

    return rx_symb_freq

def chest_cacl(rx_pilot_freq:np.ndarray, tx_pilot_ref:np.ndarray, config:OFDMConfig)->np.ndarray:
    """  
    Other CHEST CALC
    """
    tx_conj = np.conj(tx_pilot_ref)
    sqr_mag_tx = np.abs(tx_pilot_ref) ** 2
    channel_gains = (rx_pilot_freq * tx_conj) / (sqr_mag_tx + 1e-12)
    return channel_gains
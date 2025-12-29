import numpy as np
from ofdm.config import OFDMConfig

def generate_ofdm_data_symbol(config:OFDMConfig, iq_data:np.ndarray, pilot_data:np.ndarray)->np.ndarray:
    """
    Generated OFDM Symbols
    """
    #Generate Empty Symbol
    symbol_freq = np.zeros(config.N, dtype=complex)

    #Length Checks
    if len(iq_data) != len(config.data_carriers):
        raise ValueError(f"Lenth of Iq Data:{len(iq_data)} does not match length of Data Carrier Mapping: {len(config.data_carriers)} ")
    if len(pilot_data) != len(config.pilot_carriers):
        raise ValueError(f"Length of Pilot Data:{len(pilot_data)} does not match length of Pilot Carrier mapping {len(config.pilot_carriers)}")
    
    #Add IQ Data
    symbol_freq[config._idx(np.array(config.data_carriers))] = iq_data

    #Add Pilot Data
    symbol_freq[config._idx(np.array(config.pilot_carriers))] = pilot_data
    return symbol_freq

def extract_data(Rk_symbol:np.ndarray, config:OFDMConfig)->np.ndarray:
    """
    Return a ndarray of just the data subcarriers
    """
    return Rk_symbol[config.data_carriers]

def extract_pilots(Rk_symbol:np.ndarray, config:OFDMConfig)->np.ndarray:
    """
    Return a ndarray of just the pilot subcarriers
    """
    return Rk_symbol[config.pilot_carriers]


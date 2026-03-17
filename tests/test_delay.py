import numpy as np
from ofdm.channel.delay import matched_filter_calc
from ofdm.config import OFDMConfig
import pytest

def test_matched_filter_output_length():
    rx  = np.random.randn(100) + 1j * np.random.randn(100)
    ref = np.random.randn(100) + 1j * np.random.randn(100)      
    z, lags = matched_filter_calc(rx_iq=rx, ref_iq=ref,
fs=100e6)                                                       
    assert len(z) == len(rx) + len(ref) - 1
                                                                
def test_matched_filter_lags_length():                          
    rx  = np.random.randn(100) + 1j * np.random.randn(100)      
    ref = np.random.randn(100) + 1j * np.random.randn(100)      
    z, lags = matched_filter_calc(rx_iq=rx, ref_iq=ref,         
fs=100e6)                                                       
    assert len(lags) == len(z)                                  
                                                                
def test_matched_filter_peak_at_zero_delay():                   
    signal = np.random.randn(100) + 1j * np.random.randn(100)
    z, lags = matched_filter_calc(rx_iq=signal, ref_iq=signal,  
fs=100e6)                                                       
    peak_lag = lags[np.argmax(np.abs(z))]
    assert peak_lag == 0.0                     
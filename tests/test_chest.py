import numpy as np
from ofdm.channel.CHEST import channel_estimation_calc, apply_gains, chest_cacl
from ofdm.config import OFDMConfig
from ofdm.utils.generator import DataGenerator
from ofdm.core import payload, preamble
import pytest


def test_channel_estimation_unity_channel():
    config = OFDMConfig()
    pilot_symbol = preamble.generate_pilot_symbol(config=config)

    gains = channel_estimation_calc(rx_pilot_freq=pilot_symbol, tx_pilot_ref=pilot_symbol, config=config)

    active_bins = np.unique(np.concatenate([config.data_carriers, config.pilot_carriers]))
    expected = np.ones(len(active_bins), dtype=complex)
    np.testing.assert_allclose(gains[active_bins], expected, atol = 1e-10)

def test_channel_estimation_known_gains():
    config = OFDMConfig()
    pilot_symbol_tx = preamble.generate_pilot_symbol(config=config)
    pilot_symbol_rx = pilot_symbol_tx * 2

    active_bines = np.unique(np.concatenate([config.data_carriers, config.pilot_carriers]))
    gains = channel_estimation_calc(rx_pilot_freq=pilot_symbol_rx, tx_pilot_ref=pilot_symbol_tx, config=config)
    expected = np.ones(len(active_bines), dtype=complex) * 2
    np.testing.assert_allclose(gains[active_bines], expected, atol =1e-10)
    
def test_apply_gains_unity_gains_uncahnged():
    config = OFDMConfig()
    pilot_symbol_tx = preamble.generate_pilot_symbol(config=config)
    pilot_symbol_rx = pilot_symbol_tx

    active_bins = np.unique(np.concatenate([config.data_carriers, config.pilot_carriers]))
    gains = np.ones(len(active_bins), dtype=complex)
    applied_gains = apply_gains(rx_symb_freq=pilot_symbol_rx[active_bins], Lambda_est=gains)
    np.testing.assert_allclose(applied_gains,pilot_symbol_rx[active_bins], atol=1e-10)

def test_apply_gains_correct_equalization():
    config = OFDMConfig()
    pilot_symbol_tx = preamble.generate_pilot_symbol(config=config)
    active_bins = np.unique(np.concatenate([config.data_carriers, config.pilot_carriers]))                          

    channel_gain = 2.0
    pilot_symbol_rx = pilot_symbol_tx[active_bins] * channel_gain

    calced_gains = channel_estimation_calc(
        rx_pilot_freq=pilot_symbol_rx,
        tx_pilot_ref=pilot_symbol_tx[active_bins],
        config=config
    )

    equalized = apply_gains(rx_symb_freq=pilot_symbol_rx, Lambda_est=calced_gains)
    np.testing.assert_allclose(equalized, pilot_symbol_tx[active_bins], atol=1e-10)

def test_chest_calc_unity_channel():                                                                                
    config = OFDMConfig()                                                                                           
    pilot_symbol = preamble.generate_pilot_symbol(config=config)                                                    
    active_bins = np.unique(np.concatenate([config.data_carriers, config.pilot_carriers]))                          
                                                                                                                    
    gains = chest_cacl(                                                                                             
        rx_pilot_freq=pilot_symbol[active_bins],
        tx_pilot_ref=pilot_symbol[active_bins],                                                                     
        config=config
    )

    np.testing.assert_allclose(gains, np.ones(len(active_bins)), atol=1e-10)
    
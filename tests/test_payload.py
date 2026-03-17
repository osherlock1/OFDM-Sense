import numpy as np
import pytest
from ofdm.core.payload import generate_ofdm_data_symbol, extract_data, extract_pilots
from ofdm.config import OFDMConfig
from ofdm.utils.generator import DataGenerator
from ofdm.core.preamble import generate_pilot_vals

def test_generate_symbol_output_shape():
    config = OFDMConfig()
    gen = DataGenerator(config = config)
    iq_data, bits = gen.generate_random_payload()
    pilot_data = generate_pilot_vals(config=config)
    symbol = generate_ofdm_data_symbol(config=config, iq_data = iq_data, pilot_data=pilot_data)
    assert symbol.shape[0] == config.N

def test_generate_symbol_data_at_correct_bins():
    config = OFDMConfig()
    gen = DataGenerator(config = config)
    iq_data, bits = gen.generate_random_payload()
    pilot_data = generate_pilot_vals(config=config)
    symbol = generate_ofdm_data_symbol(config=config, iq_data=iq_data, pilot_data=pilot_data)
    np.testing.assert_array_equal(symbol[config.data_carriers], iq_data)


def test_generate_symbol_pilots_at_correct_bins():
    config = OFDMConfig()
    gen = DataGenerator(config = config)
    iq_data, bits = gen.generate_random_payload()
    pilot_data = generate_pilot_vals(config=config)
    symbol = generate_ofdm_data_symbol(config=config, iq_data=iq_data, pilot_data=pilot_data)
    np.testing.assert_array_equal(symbol[config.pilot_carriers], pilot_data)

def test_generate_symbol_guard_is_zero():
    config = OFDMConfig()
    gen = DataGenerator(config = config)
    iq_data = np.zeros(len(config.data_carriers), dtype=complex)
    pilot_data = np.zeros(len(config.pilot_carriers), dtype=complex)
    symbol = generate_ofdm_data_symbol(config=config, iq_data=iq_data, pilot_data=pilot_data)
    
    half = config.N // 2
    guard_indicies = list(range(half - config.GUARD_LEN, half + config.GUARD_LEN + 1))
    guard_indicies.append(0)

    np.testing.assert_array_equal(symbol[guard_indicies], 0)

def test_generate_symbol_wrong_iq_length_raises():
    config = OFDMConfig()
    iq_data = np.zeros(3, dtype=complex)
    pilot_data = generate_pilot_vals(config=config)
    with pytest.raises(ValueError):
        symbol = generate_ofdm_data_symbol(config=config, iq_data = iq_data, pilot_data=pilot_data)

def test_generate_symbol_wrong_pilot_length_raises():
    config = OFDMConfig()
    iq_data = np.zeros(len(config.data_carriers), dtype=complex)
    pilot_data = np.zeros(3, dtype=complex)
    with pytest.raises(ValueError):
        generate_ofdm_data_symbol(config=config, iq_data=iq_data, pilot_data=pilot_data)

def test_extract_data_length():
    config = OFDMConfig()
    iq_data = np.zeros(len(config.data_carriers), dtype=complex)
    pilot_data = np.zeros(len(config.pilot_carriers), dtype=complex)
    symbol = generate_ofdm_data_symbol(config=config, iq_data=iq_data, pilot_data=pilot_data)
    recovered_iq_data = extract_data(symbol, config=config)
    assert len(recovered_iq_data) == len(config.data_carriers)

def test_extract_pilots_length():
    config = OFDMConfig()
    iq_data = np.zeros(len(config.data_carriers), dtype=complex)
    pilot_data = np.zeros(len(config.pilot_carriers), dtype=complex)
    symbol = generate_ofdm_data_symbol(config=config, iq_data=iq_data, pilot_data=pilot_data)
    recovered_pilot_data = extract_pilots(symbol, config=config)
    assert len(recovered_pilot_data) == len(config.pilot_carriers)

def test_extract_data_correct_values():
    config = OFDMConfig()
    gen = DataGenerator(config=config)
    iq_data, bits = gen.generate_random_payload()
    pilot_data = np.zeros(len(config.pilot_carriers), dtype=complex)
    symbol = generate_ofdm_data_symbol(config=config, iq_data=iq_data, pilot_data=pilot_data)
    recovered_iq_data = extract_data(symbol, config=config)
    np.testing.assert_array_equal(recovered_iq_data, iq_data)


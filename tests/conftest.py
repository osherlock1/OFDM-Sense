import numpy as np
import pytest

from ofdm.config import OFDMConfig
from ofdm.utils.generator import DataGenerator
from ofdm.core.preamble import generate_pilot_symbol, generate_pilot_vals
from ofdm.core.payload import generate_ofdm_data_symbol
from ofdm.core.waveform import create_time_domain_symbol


@pytest.fixture(scope="session")
def cfg():
    return OFDMConfig(seed=42)


@pytest.fixture(scope="session")
def active_bins(cfg):
    return np.unique(np.concatenate([cfg.data_carriers], cfg.pilot_carriers))


@pytest.fixture(scope="session")
def pilot_values(cfg):
    return generate_pilot_vals(config=cfg)


@pytest.fixture(scope="session")
def pilot_symbol_freq(cfg):
    return generate_pilot_symbol(config=cfg)


@pytest.fixture(scope="session")
def data_symbol_freq(cfg, pilot_vals):
    gen = DataGenerator(config=cfg, seed=0)
    iq_tx, bits_tx = gen.generate_random_payload()
    symbol = generate_ofdm_data_symbol(config=cfg, iq_data=iq_tx, pilot_data=pilot_vals)
    return symbol, iq_tx, bits_tx


@pytest.fixture(scope="session")
def pilot_symbol_time(cfg, pilot_symbol_freq):
    return create_time_domain_symbol(pilot_symbol_freq, cp_len=cfg.CP_LEN)


@pytest.fixture(scope="session")
def data_symbol_time(cfg, data_symbol_freq):
    return create_time_domain_symbol(data_symbol_freq, cp_len=cfg.CP_LEN)


@pytest.fixture
def generator(cfg):
    return DataGenerator(config=cfg, seed=0)

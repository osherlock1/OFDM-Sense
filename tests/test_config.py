import numpy as np
import pytest
from ofdm.config import OFDMConfig

def test_default_config_creates_without_error():
    config = OFDMConfig()
    assert isinstance(config, OFDMConfig)

def test_pilot_and_data_carriers_no_overlap():
    config = OFDMConfig()
    data_carriers_set = set(config.data_carriers)
    pilot_carriers_set = set(config.pilot_carriers)
    intersection = data_carriers_set.intersection(pilot_carriers_set)
    assert not intersection

def test_all_carriers_within_valid_range():
    config = OFDMConfig()
    all_carriers = config.data_carriers + config.pilot_carriers
    assert all(0 <= idx <= config.N - 1 for idx in all_carriers)

def test_dc_carrier_not_allowed():
    config = OFDMConfig()
    all_carriers = config.data_carriers + config.pilot_carriers
    assert any((idx != 0) for idx in all_carriers)

def test_gaurd_band_not_allocated():
    config = OFDMConfig()
    half = config.N // 2
    valid_pos_k = range(1, half - config.GUARD_LEN)
    valid_neg_k = range(-half + config.GUARD_LEN, 0)
    valid_indicies = set(config._idx(k) for k in list(valid_pos_k) + list(valid_neg_k))
    
    all_carriers = set(config.data_carriers + config.pilot_carriers)
    assert all_carriers.issubset(valid_indicies)

def test_carrier_count_matches_expected():
    config = OFDMConfig()
    expected_pilots = config.N // 4
    expected_data = len(config.data_carriers + config.pilot_carriers) - expected_pilots

    assert len(config.pilot_carriers) == expected_pilots
    assert len(config.data_carriers) == expected_data

def test_duplicate_pilot_data_raises():
    with pytest.raises(ValueError):
        OFDMConfig(
            data_carriers=[0, 1, 2, 3],
            pilot_carriers=[3, 4, 5, 6]
        )

def test_reproducibility():
    config1 = OFDMConfig()
    config2 = OFDMConfig()
    assert config1.data_carriers == config2.data_carriers
    assert config1.pilot_carriers == config2.pilot_carriers
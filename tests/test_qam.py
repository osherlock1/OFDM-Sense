import numpy as np
from ofdm.modulation.qam import binary_to_iq, iq_to_binary, get_reference_constalation
import pytest


@pytest.mark.parametrize("bits,expected_real,expected_imag", [
    ("0000", -3, -3),
    ("1111",  1,  1),
    ("1010",  3,  3),
])

def test_binary_to_iq_known_values(bits, expected_real, expected_imag):
    scale = np.sqrt(10)
    result = binary_to_iq(bits)
    assert abs(result.real - expected_real / scale) < 1e-10
    assert abs(result.imag - expected_imag / scale) < 1e-10

def test_binary_to_iq_wrong_length_raises():
    """Should raise ValueError for inputs that arent 4 bits"""
    with pytest.raises(ValueError):
        binary_to_iq("101")

def test_iq_roundtrip():
    """Encode then decode should recover original bits"""
    for bits in ["0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111", "1000", "1001", "1010", "1011", "1100", "1101", "1110", "1111"]:
        iq = binary_to_iq(bits)
        scale = np.sqrt(10)
        recovered = iq_to_binary(iq * scale)
        assert recovered == bits
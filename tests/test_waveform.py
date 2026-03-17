import numpy as np
from ofdm.core.waveform import add_cp, remove_cp, freq_to_time, time_to_freq


def test_add_cp_length():
    """Output should be len(signal) + cp_len"""
    signal = np.ones(256, dtype=complex)
    result = add_cp(signal, cp_len=64)
    assert len(result) == 320

def test_add_cp_content():
    """The prefix should be the last cp_len samples of the original signal"""
    signal = np.arange(256, dtype=complex)
    result = add_cp(signal, cp_len = 64)
    np.testing.assert_array_equal(result[:64], signal[-64:])

def test_remove_cp_is_inverse_of_add_cp():
    """add_cp then remove_cp should recover the origin signal"""
    original = np.random.randn(256) + 1j * np.random.randn(256)
    with_cp = add_cp(original, cp_len=64)
    recovered = remove_cp(with_cp, cp_len=64)
    np.testing.assert_array_equal(recovered, original)

def test_add_cp_zero_length():
    """cp_len = 0 should return the signal unchanged"""
    signal = np.ones(256, dtype=complex)
    result = add_cp(signal, cp_len=0)
    np.testing.assert_array_equal(result, signal)

def test_freq_time_roundtrip():
    """IFFT then FFT should recover the original"""
    freq = np.random.randn(256) + 1j * np.random.randn(256)
    recovered = time_to_freq(freq_to_time(freq))
    np.testing.assert_allclose(recovered, freq, atol=1e-10)
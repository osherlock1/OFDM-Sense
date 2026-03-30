import numpy as np
from ofdm.channel.delay import matched_filter_calc, upsample, scale_rx_signal
from ofdm.config import OFDMConfig
import pytest
from scripts.delay.calc_delay import calculate_precise_delay


class TestUpsample:
    def test_output_length(self):
        x = np.ones(10, dtype=complex)
        out = upsample(x, scale_factor=4)
        assert len(out) == 40
    
    def test_scale_factor_one_unchanged(self):
        x = np.random.randn(64) + 1j * np.random.randn(64)
        out = upsample(x, scale_factor=1)
        np.testing.assert_allclose(np.abs(out), np.abs(x), atol=1e-10)

    def test_pure_tone_preserved(self):
        N = 64
        f = 5
        t = np.arange(N)
        x = np.exp(1j * 2 * np.pi * f * t / N)
        out = upsample(x, scale_factor=4)
        assert len(out) == N * 4

class TestScaleRxSignal:
    def test_max_value_is_09(self):
        x = np.array([0.5, 1.0, -2.0, 0.3], dtype=float)
        out = scale_rx_signal(x)
        assert np.max(np.abs(out)) == pytest.approx(0.9)

    def test_zero_signal_unchanged(self):
        x = np.zeros(10)
        out = scale_rx_signal(x)
        np.testing.assert_array_equal(out, x)

class TestCalculatePreciseDelay:
    def test_zero_delay(self):
        ref = np.random.randn(64) + 1j * np.random.rand(64)
        delay = calculate_precise_delay(ref, ref, fs=100e6)
        assert abs(delay) < 1 / 100e6

    def test_known_integer_delay(self):
        ref=np.zeros(256, dtype=complex)
        ref[10:20] = 1.0
        rx = np.zeros(256, dtype=complex)
        rx[20:30] = 1.0
        delay = calculate_precise_delay(rx, ref, fs=100e6)
        expected = 10 / 100e6
        assert abs(delay - expected) < 2 / 100e6

    def test_returns_scaler(self):
        ref = np.random.randn(64) + 1j * np.random.randn(64)
        result = calculate_precise_delay(ref, ref, fs=100e6)
        assert np.isscalar(result) or result.ndim == 0

def test_matched_filter_output_length():
    rx  = np.random.randn(100) + 1j * np.random.randn(100)
    ref = np.random.randn(100) + 1j * np.random.randn(100)      
    z, lags = matched_filter_calc(rx_iq=rx, ref_iq=ref, fs=100e6)                                                       
    assert len(z) == len(rx) + len(ref) - 1
                                                                
def test_matched_filter_lags_length():                          
    rx  = np.random.randn(100) + 1j * np.random.randn(100)      
    ref = np.random.randn(100) + 1j * np.random.randn(100)      
    z, lags = matched_filter_calc(rx_iq=rx, ref_iq=ref, fs=100e6)                                                       
    assert len(lags) == len(z)                                  
                                                                
def test_matched_filter_peak_at_zero_delay():                   
    signal = np.random.randn(100) + 1j * np.random.randn(100)
    z, lags = matched_filter_calc(rx_iq=signal, ref_iq=signal, fs=100e6)                                                       
    peak_lag = lags[np.argmax(np.abs(z))]
    assert peak_lag == 0.0                     
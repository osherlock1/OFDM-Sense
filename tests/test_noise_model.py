import numpy as np
import pytest

from ofdm.simulation.noise_model import add_gaussian_noise

def test_output_shape():
    tdoa = np.array([1e-9, 2e-9, 3e-9])
    noisy = add_gaussian_noise(tdoa, sigma_ns=2.0)
    assert noisy.shape == tdoa.shape

def test_noise_is_applied():
    tdoa = np.array([1e-9, 2e-9, 3e-9])
    noisy = add_gaussian_noise(tdoa, sigma_ns=2.0, rng=np.random.default_rng(42))
    assert not np.allclose(noisy, tdoa, atol=1e-12)
    
def test_reproducible_with_seed():
    tdoa = np.array([1e-9, 2e-9, 3e-9])
    noisy = add_gaussian_noise(tdoa, sigma_ns=2.0, rng=np.random.default_rng(0))
    noisy2 = add_gaussian_noise(tdoa,sigma_ns=2.0, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(noisy, noisy2)

def test_zero_noise_unchanged():
    tdoa = np.array([1e-9, 2e-9, 3e-9])
    noisy = add_gaussian_noise(tdoa, sigma_ns=0)
    np.testing.assert_array_equal(noisy, tdoa)
import numpy as np
from ofdm.config import OFDMConfig
from ofdm.core import preamble
from ofdm.channel.noise import add_noise


def test_add_noise_output_shape():
    config = OFDMConfig()
    pilot_symbol = preamble.generate_pilot_symbol(config=config)
    noisy_signal = add_noise(signal=pilot_symbol, snr_db=8)
    assert len(pilot_symbol) == len(noisy_signal)


def test_add_noise_is_not_identical_to_input():
    config = OFDMConfig()
    pilot_symbol = preamble.generate_pilot_symbol(config=config)
    noisy_signal = add_noise(signal=pilot_symbol, snr_db=8)
    assert not np.array_equal(noisy_signal, pilot_symbol)


def test_add_noise_reproducible_with_seed():
    config = OFDMConfig()
    pilot_symbol = preamble.generate_pilot_symbol(config=config)
    noisy_signal1 = add_noise(signal=pilot_symbol, snr_db=8, seed=42)
    noisy_signal2 = add_noise(signal=pilot_symbol, snr_db=8, seed=42)
    np.testing.assert_array_equal(noisy_signal1, noisy_signal2)


def test_add_noise_different_seeds_differ():
    config = OFDMConfig()
    pilot_symbol = preamble.generate_pilot_symbol(config=config)
    noisy_signal1 = add_noise(signal=pilot_symbol, snr_db=42, seed=42)
    noisy_signal2 = add_noise(signal=pilot_symbol, snr_db=8, seed=42)
    assert not np.array_equal(noisy_signal1, noisy_signal2)


def test_add_noise_snr_approximate():
    pilot_symbol = preamble.generate_pilot_symbol(config=OFDMConfig())
    target_snr_db = 30.0
    noisy_signal = add_noise(signal=pilot_symbol, snr_db=target_snr_db, seed=42)

    noise = noisy_signal - pilot_symbol
    signal_power = np.mean(np.abs(pilot_symbol) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)
    measured_snr_db = 10 * np.log10(signal_power / noise_power)

    assert abs(measured_snr_db - target_snr_db) < 1.0

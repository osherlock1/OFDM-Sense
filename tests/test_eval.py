import numpy as np
import math
from ofdm.utils.eval import calc_BER, calc_EVM, calc_SER


def test_evm_identical_signals_is_negative_inf():
    signal = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    evm = calc_EVM(iq_rx=signal, iq_ref=signal)
    assert math.isinf(evm) and evm < 0


def test_evm_known_error():
    iq_ref = np.array([2 + 0j])
    iq_rx = np.array([3 + 0j])
    evm = calc_EVM(iq_rx=iq_rx, iq_ref=iq_ref)
    assert abs(evm - (-6.0206)) < 0.001


def test_evm_mismatched_length_truncates():
    iq_ref = np.array([1 + 0j, 2 + 0j, 3 + 0j])
    iq_rx = np.array([1 + 0j, 2 + 0j])
    calc_EVM(iq_rx=iq_rx, iq_ref=iq_ref)


def test_ser_identical_signals_is_zero():
    signal = np.array([complex(-3, -3), complex(1, 1), complex(3, -1)])
    ser = calc_SER(iq_rx=signal, iq_ref=signal)
    assert ser == 0.0


def test_ser_all_wrong_is_one():
    iq_ref = np.array([complex(-3, -3)])
    iq_rx = np.array([complex(3, 3)])
    ser = calc_SER(iq_rx=iq_rx, iq_ref=iq_ref)
    assert ser == 1.0


def test_ber_identical_signals_is_zero():
    signal = np.array([complex(-3, -3), complex(1, 1), complex(3, -1)])
    ber = calc_BER(iq_rx=signal, iq_ref=signal)
    assert ber == 0.0


def test_ber_all_bits_flipped_is_one():
    iq_ref = np.array([complex(-3, -3)])
    iq_rx = np.array([complex(1, 1)])
    ber = calc_BER(iq_rx=iq_rx, iq_ref=iq_ref)
    assert ber == 1.0


def test_ber_mismatched_length_truncates():
    iq_ref = np.array([complex(-3, -3), complex(1, 1), complex(3, -1)])
    iq_rx = np.array([complex(-3, -3), complex(1, 1)])
    calc_BER(iq_rx=iq_rx, iq_ref=iq_ref)

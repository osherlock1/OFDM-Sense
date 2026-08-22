from ofdm.config import OFDMConfig
from ofdm.utils.generator import DataGenerator


def test_generate_payload_iq_length():
    config = OFDMConfig()
    gen = DataGenerator(config=config)
    iq_data, bits = gen.generate_random_payload()
    assert len(iq_data) == len(config.data_carriers)


def test_generate_payload_bits_length():
    config = OFDMConfig()
    gen = DataGenerator(config=config)
    iq_data, bits = gen.generate_random_payload()
    assert len(bits) == len(config.data_carriers) * 4


def test_generate_payload_reproducible_with_seed():
    config = OFDMConfig()
    gen1 = DataGenerator(config=config, seed=7)
    _, bits1 = gen1.generate_random_payload()
    gen2 = DataGenerator(config=config, seed=7)
    _, bits2 = gen2.generate_random_payload()
    assert bits1 == bits2


def test_generate_payload_bits_are_binary():
    config = OFDMConfig()
    gen = DataGenerator(config=config)
    _, bits = gen.generate_random_payload()
    assert all(b in ("0", "1") for b in bits)

import numpy as np
from ofdm.modulation.qam import binary_to_iq
from ofdm.config import OFDMConfig
from ofdm.core import preamble, payload, waveform
import json
from PIL import Image
import io
from ofdm.channel.noise import add_noise


IMAGE_PATH = "./images/cat.jpeg"
SIGNAL_SAVE_PATH = "./data_files/image.dat"
METADATA_SAVE_PATH = "./metadata/image_metadata.json"
OFDM_CONF = OFDMConfig()
SYMBOLS_PER_PACKET = 100


def bytes2bits(data_bytes: bytes):
    bits = "".join(format(b, "08b") for b in data_bytes)
    return bits


def pad_and_reformat_iq_data(iq_data: np.ndarray) -> np.ndarray:
    n_data_subcarriers = len(OFDM_CONF.data_carriers)
    remainder = len(iq_data) % n_data_subcarriers
    if remainder != 0:
        pad = n_data_subcarriers - remainder
        iq_samples = np.concatenate([iq_data, np.zeros(pad, dtype=complex)])
    return iq_samples.reshape(-1, n_data_subcarriers)


def main():
    img = Image.open(IMAGE_PATH)
    buf = io.BytesIO()
    img.save(buf, format='BMP')
    image_bytes = buf.getvalue()

    bits = bytes2bits(image_bytes)
    chunks4bit = [bits[i:i + 4] for i in range(0, len(bits), 4)]
    iq_data = np.array([binary_to_iq(chunk) for chunk in chunks4bit])

    symbol_samples = pad_and_reformat_iq_data(iq_data)

    data_symbols_td = []
    for symbol_iq in symbol_samples:
        pilot_vals = preamble.generate_pilot_vals(config=OFDM_CONF)
        data_symbol_freq = payload.generate_ofdm_data_symbol(config=OFDM_CONF, iq_data=symbol_iq, pilot_data=pilot_vals)
        data_symbol_final = waveform.create_time_domain_symbol(data_symbol_freq, cp_len=OFDM_CONF.CP_LEN)
        data_symbols_td.append(data_symbol_final)

    packets = []
    for i in range(0, len(data_symbols_td), SYMBOLS_PER_PACKET):
        chunk = data_symbols_td[i: i + SYMBOLS_PER_PACKET]
        packets.append(chunk)

    all_segments = [np.zeros(20000, dtype=complex)]
    inter_gap = np.zeros(5000, dtype=complex)
    for i, chunk in enumerate(packets):
        sync_td = waveform.create_time_domain_symbol(preamble.generate_sync_symbol(OFDM_CONF), OFDM_CONF.CP_LEN)
        pilot_td = waveform.create_time_domain_symbol(preamble.generate_pilot_symbol(OFDM_CONF), OFDM_CONF.CP_LEN)
        ofdm_packet = np.concatenate([sync_td, pilot_td] + chunk)
        all_segments.append(ofdm_packet)
        if i < len(packets) - 1:
            all_segments.append(inter_gap)

    all_segments.append(np.zeros(20000, dtype=complex))
    final_signal = np.concatenate(all_segments)

    max_val = np.max(np.abs(final_signal))
    if max_val > 0:
        final_signal = final_signal * (0.9 / max_val)

    final_signal = add_noise(final_signal, 100)

    final_signal.astype(np.complex64).tofile(SIGNAL_SAVE_PATH)

    ref_data = {
        "n_samples": len(final_signal),
        "symb_per_packet": SYMBOLS_PER_PACKET,
        "total_image_bytes": len(image_bytes),
        "n_packets": len(packets)
    }
    with open(METADATA_SAVE_PATH, 'w') as f:
        json.dump(ref_data, f, indent=2)


if __name__ == "__main__":
    main()

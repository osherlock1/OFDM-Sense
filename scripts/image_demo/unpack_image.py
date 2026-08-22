import numpy as np
from ofdm.processing.rx import normalize_rx_signal, unpack_rx_packet
from ofdm.config import OFDMConfig
from ofdm.core import sync, waveform, preamble
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from ofdm.modulation.qam import iq_to_binary
import json
from PIL import Image
import io
from tqdm import tqdm


IMAGE_RX_PATH = "data_files/image_rx.00.dat"
IMAGE_PATH = "images/cat.jpeg"
OUTPUT_IMAGE_PATH = "images/output_image.bmp"
OFDM_CONF = OFDMConfig()
IMAGE_METADATA_PATH = "metadata/image_metadata.json"


def main():
    with open(IMAGE_METADATA_PATH, "r") as f:
        image_metadata = json.load(f)
    n_data_symbols_per_packet = image_metadata['symb_per_packet']

    raw_data = np.fromfile(IMAGE_RX_PATH, dtype=np.complex64)
    rx_data = normalize_rx_signal(raw_data)

    M, P = sync.calculate_schmidl_cox_metrics(rx_signal=rx_data, config=OFDM_CONF)
    peaks, _ = find_peaks(M, height=0.75, distance=10000)

    ofdm_packet_len = (n_data_symbols_per_packet + 2) * (OFDM_CONF.N + OFDM_CONF.CP_LEN) + 500

    sliced_packets = []
    for peak in peaks:
        start_idx = max(0, peak - 500)
        end_idx = min(len(rx_data) - 1, peak + ofdm_packet_len)
        sliced_packets.append(rx_data[start_idx:end_idx])
    sliced_packets = sliced_packets[3:-1]

    sync_td = waveform.create_time_domain_symbol(preamble.generate_sync_symbol(OFDM_CONF), OFDM_CONF.CP_LEN)
    pilot_td = waveform.create_time_domain_symbol(preamble.generate_pilot_symbol(OFDM_CONF), OFDM_CONF.CP_LEN)

    demodulated_data_list = []
    for packet in tqdm(sliced_packets, desc="Unpacking OFDM Packets"):
        demodulated_data, refined_packet_start = unpack_rx_packet(
            ofdm_conf=OFDM_CONF,
            rx_raw=packet,
            sync_ref_time=sync_td,
            tx_pilot_ref=pilot_td,
            n_data_symbols=n_data_symbols_per_packet
        )
        demodulated_data_list.append(demodulated_data)

    n_packets = len(demodulated_data_list)
    cols = 4
    rows = (n_packets + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()

    for i, pkt_iq in enumerate(demodulated_data_list):
        axes[i].scatter(pkt_iq.real, pkt_iq.imag, s=1, alpha=0.3)
        axes[i].set_title(f"Packet {i}")
        axes[i].set_xlim(-5, 5)
        axes[i].set_ylim(-5, 5)
        axes[i].axhline(0, color='k', linewidth=0.5)
        axes[i].axvline(0, color='k', linewidth=0.5)

    for j in range(n_packets, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("images/packet_constellations.png")
    plt.show()

    all_iq = np.concatenate(demodulated_data_list)
    all_iq = all_iq * np.sqrt(10)
    print(f"IQ sample range: min={np.min(np.abs(all_iq)):.3f}, max={np.max(np.abs(all_iq)):.3f}")
    print(f"First 5 IQ samples: {all_iq[:5]}")
    bits = "".join(iq_to_binary(sample) for sample in all_iq)

    n_bytes = len(bits) // 8
    bits = bits[:n_bytes * 8]
    image_bytes = bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))

    magnitudes = np.abs(all_iq)
    print(f"min:    {np.min(magnitudes):.3f}")
    print(f"max:    {np.max(magnitudes):.3f}")
    print(f"mean:   {np.mean(magnitudes):.3f}")
    print(f"median: {np.median(magnitudes):.3f}")
    print(f"95th percentile: {np.percentile(magnitudes, 95):.3f}")

    img = Image.open(IMAGE_PATH)
    buf = io.BytesIO()
    img.save(buf, format='BMP')
    original = buf.getvalue()

    matches = sum(a == b for a, b in zip(image_bytes, original))
    print(f"Matching bytes: {matches}/{len(original)} ({100*matches/len(original):.1f}%)")

    bytes_per_packet = (n_data_symbols_per_packet * len(OFDM_CONF.data_carriers) * 4) // 8

    for i, pkt_iq in enumerate(demodulated_data_list):
        pkt_iq_scaled = pkt_iq * np.sqrt(10)
        pkt_bits = "".join(iq_to_binary(s) for s in pkt_iq_scaled)
        pkt_bytes = bytes(int(pkt_bits[j:j+8], 2) for j in range(0, len(pkt_bits)//8*8, 8))

        orig_slice = original[i * bytes_per_packet: (i+1) * bytes_per_packet]
        matches = sum(a == b for a, b in zip(pkt_bytes, orig_slice))
        print(f"Packet {i}: {matches}/{len(orig_slice)} ({100*matches/max(len(orig_slice),1):.1f}%)")

    image_bytes_fixed = original[:54] + image_bytes[54:]

    with open(OUTPUT_IMAGE_PATH, "wb") as f:
        f.write(image_bytes_fixed)

    print(f"Saved to {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()

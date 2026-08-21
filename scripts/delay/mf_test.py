import numpy as np
import matplotlib.pyplot as plt
import json
from ofdm.channel import delay
from ofdm.modulation import qam
from ofdm.config import OFDMConfig


# =========================
# Main
# =========================
def main():

    # Define OFDM Configuration
    ofdm_conf = OFDMConfig()

    # Unpack Raw RX and TX data
    raw_rx_data = np.fromfile("data_files/rand_ofdm_packet_rx.dat", dtype=np.complex64)
    raw_tx_data = np.fromfile("data_files/rand_ofdm_packet.dat", dtype=np.complex64)

    # Unpack TX Pilot symbol
    with open("data_files/rand_ofdm_packet_ref.json", "r") as f:
        ref_data = json.load(f)

    # Get Tx pilot symbol
    tx_pilot = np.array(ref_data["pilot_ref_real"]) + 1j * np.array(
        ref_data["pilot_ref_imag"]
    )

    # Scale raw_rx_data
    max_val = np.max(np.abs(raw_rx_data))

    if max_val > 0:
        scale_factor = 0.9 / max_val
        raw_rx_data = raw_rx_data * scale_factor

    # Calculate Matched Filter Delay
    z, lags = delay.matched_filter_calc(
        rx_iq=raw_rx_data, ref_iq=tx_pilot, fs=ofdm_conf.FS
    )
    z_mag = np.abs(z)
    # Find Peak of correlation
    peak_idx = np.argmax(z_mag)

    # Interpolation for fine resoluation
    K = 100
    N = len(z)
    N_padded = N * K

    # Freq Domain
    Z_freq = np.fft.fft(z)

    pivot = (N + 1) // 2

    # Zero pad
    Z_padded = np.zeros(N_padded, dtype=np.complex64)
    Z_padded[:pivot] = Z_freq[:pivot]
    Z_padded[-(N - pivot) :] = Z_freq[-(N - pivot) :]

    # Move pack to time domain
    z_upsampled = np.fft.ifft(Z_padded) * K

    # High res lags
    lags_upsampled = np.linspace(lags[0], lags[-1], N_padded)

    z_mag_up = np.abs(z_upsampled)
    fine_peak_idx = np.argmax(z_mag_up)
    fine_delay = lags_upsampled[fine_peak_idx]

    # Calculate Distance
    SPEED_OF_LIGHT = 299792458
    CONSTANT = -19595.82587042264
    raw_distance = (fine_delay * SPEED_OF_LIGHT) - CONSTANT

    # Calibrate
    actual_distance = 1  # 1 meter

    constant = (fine_delay * SPEED_OF_LIGHT) - actual_distance
    print(f"Calculated Constant: {constant}")

    print(f"Coarse Delay: {lags[peak_idx] * 1e6:.4f}us")
    print(f"Fine Delay: {fine_delay * 1e6:.4f}us")
    print(f"Distance: {raw_distance:.2f}meters")

    plt.figure()
    plt.plot(lags_upsampled, np.abs(z_mag_up))
    plt.show()


def unpack_json(json_file_name: str) -> dict:
    """
    Unpacks a json file and returns the json dictionary

    Args:
        json_file_name: name of file in the data_files dir (do not include data_files/)

    Returns:
        json data dictionary
    """
    json_file_path = f"data_files/{json_file_name}"
    with open(json_file_path, "r") as f:
        data = json.load(f)
    print(f"[Success] Unpacked {json_file_path}")
    return data


def binary_ref_to_iq(binary_string: str, n_samples: int) -> np.ndarray:

    full_string = "".join(binary_string)

    # Parse String into 4 bit words
    word_len = 4
    binary_word_list = np.array(
        [full_string[i : word_len + i] for i in range(0, len(full_string), word_len)]
    )

    # Convert to IQ
    iq_array = [qam.binary_to_iq(word) for word in binary_word_list]
    return np.array(iq_array) * np.sqrt(10)


if __name__ == "__main__":
    main()

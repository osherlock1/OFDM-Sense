import numpy as np
import os
import json
from ofdm.channel import delay
from ofdm.config import OFDMConfig
import datetime
import scipy
import argparse
from ofdm.utils import usrp

# Config
CALIBRATION_PATH = "metadata/calibration.json"
RX_DATA_PATH = "data_files/rand_ofdm_packet_rx.01.dat"
WIRED_DATA_PATH = "data_files/rand_ofdm_packet_rx.00.dat"
TX_REF_PATH = "data_files/rand_ofdm_packet_ref.json"
ofdm_conf = OFDMConfig()
# C = scipy.constants.c #Speed of light
C = 299792458
REFERENCE_DISTANCE = 1.1  # 1 Meter reference
USRP_CONFIG_PATH = "./configs/usrp_settings.yaml"
START_IDX_PATH = "./data_files/ofdm_performance.json"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref",
        action="store_true",
        help="Declare if a direct wired connection is being used (wired ref = true, no ref = false)",
    )

    args = parser.parse_args()
    print(args.ref)

    if args.ref == True:
        rx_data_path = (
            "data_files/rand_ofdm_packet_rx.01.dat"  # FIXME: HARD CODED ADRESSED
        )
        rx_ref_path = "data_files/rand_ofdm_packet_rx.00.dat"

        # Unpack Sivers RX-
        raw_rx_data = np.fromfile(rx_data_path, dtype=np.complex64)

        # Unpack Wired Ref RX
        wired_rx_data = np.fromfile(rx_ref_path, dtype=np.complex64)

        # Unpack TX pilot symbol
        with open(TX_REF_PATH, "r") as f:
            ref_data = json.load(f)

        calibrate_with_ref(
            ref_data=ref_data, raw_rx_data=raw_rx_data, wired_rx_data=wired_rx_data
        )

    else:
        # Load Configurations
        usrp_conf = usrp.load_config(USRP_CONFIG_PATH)
        rx_channel_idx = usrp_conf.rx_channel_idx.replace(",", "")
        print(f"Unpacking {len(rx_channel_idx)} RX Channels...")

        with open(TX_REF_PATH, "r") as f:
            ref_data = json.load(f)

        with open(START_IDX_PATH, "r") as f:
            ofdm_performance_data = json.load(f)
        start_idx_list = ofdm_performance_data["start_idx"]

        constant_list = []
        for i, channel in enumerate(rx_channel_idx):
            print(f"Calibrating Channel {channel}")
            rx_data_path = f"data_files/rand_ofdm_packet_rx.0{channel}.dat"
            raw_rx_data = np.fromfile(rx_data_path, dtype=np.complex64)
            current_start_idx = start_idx_list[i]
            cleaned_rx_data = clean_rx(raw_rx_data, current_start_idx[1])

            constant = calibrate_no_ref(raw_rx_data=cleaned_rx_data, ref_data=ref_data)
            constant_list.append(constant)

        # Save Constant to JSON
        json_data = {
            "reference_distance": REFERENCE_DISTANCE,
            "constants": constant_list,
            "calibration_time": datetime.datetime.now().isoformat(),
            "mode": "no wired reference",
        }

        os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
        with open(CALIBRATION_PATH, "w") as f:
            json.dump(json_data, f, indent=2)
        print(
            f"[Success] Calibrations with wired reference finished.  Saved constant to {CALIBRATION_PATH}"
        )


def clean_rx(rx_raw: np.ndarray, start_idx: int) -> np.ndarray:
    """
    Removes leading and trialing zeros from the signal
    """

    # ------ Prepare RX signal --------
    with open(TX_REF_PATH, "r") as f:
        ref_data = json.load(f)

    # Get total samples in packet
    sym_len = ofdm_conf.N + ofdm_conf.CP_LEN
    total_symbols = 1 + 1 + ref_data["n_data_symb"]
    total_samples = sym_len * total_symbols

    buffer = 0
    start = int(start_idx - buffer)
    end = int(start_idx + total_samples + buffer)

    if start < 0:
        start = 0
    if end > len(rx_raw):
        end = len(rx_raw) - 1
    return rx_raw[start:end]


def calibrate_with_ref(
    ref_data: np.ndarray, raw_rx_data: np.ndarray, wired_rx_data: np.ndarray
):

    # Get Tx Pilot symbol
    tx_pilot = np.array(ref_data["pilot_ref_real"]) + 1j * np.array(
        ref_data["pilot_ref_imag"]
    )

    # Scale raw_rx_data
    scaled_wireless_rx = scale_rx_signal(raw_rx_data=raw_rx_data)
    scaled_wired_rx = scale_rx_signal(raw_rx_data=wired_rx_data)

    # Upsample Data
    rx_wireless_upsampled = upsample(scaled_wireless_rx, scale_factor=100)
    rx_wired_upsampled = upsample(scaled_wired_rx, scale_factor=100)
    tx_upsampled = upsample(tx_pilot, scale_factor=100)

    # Calculate Matched Filter Delay for SIVERS
    z_wireless, lags_wireless = delay.matched_filter_calc(
        rx_iq=rx_wireless_upsampled, ref_iq=tx_upsampled, fs=(ofdm_conf.FS * 100)
    )
    z_mag_wireless = np.abs(z_wireless)

    # Find Peak
    peak_idx_wireless = np.argmax(z_mag_wireless)
    fine_delay_wireless = lags_wireless[peak_idx_wireless]

    # Calculate Matched Filter Delay for WIRED
    z_wired, lags_wired = delay.matched_filter_calc(
        rx_iq=rx_wired_upsampled, ref_iq=tx_upsampled, fs=(ofdm_conf.FS * 100)
    )
    z_mag_wired = np.abs(z_wired)

    # Find Peak
    peak_idx_wired = np.argmax(z_mag_wired)
    fine_delay_wired = lags_wireless[peak_idx_wired]

    # Calculate Calibration Constant
    constant = ((fine_delay_wireless - fine_delay_wired) * C) - REFERENCE_DISTANCE
    print(f"Calculated Constant: {constant}")

    # Save Constant to JSON
    json_data = {
        "reference_distance": REFERENCE_DISTANCE,
        "constant": constant,
        "calibration_time": datetime.datetime.now().isoformat(),
        "mode": "with wired reference",
    }

    os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(json_data, f, indent=2)
    print(
        f"[Success] Calibration with wired reference finished.  Saved constant to {CALIBRATION_PATH}"
    )


def calibrate_no_ref(raw_rx_data: np.ndarray, ref_data: np.ndarray):

    # Get Tx Pilot symbol
    tx_pilot = np.array(ref_data["pilot_ref_real"]) + 1j * np.array(
        ref_data["pilot_ref_imag"]
    )

    # Scale raw_rx_data
    scaled_wireless_rx = scale_rx_signal(raw_rx_data=raw_rx_data)

    # Upsample Data
    rx_wireless_upsampled = upsample(scaled_wireless_rx, scale_factor=100)
    tx_upsampled = upsample(tx_pilot, scale_factor=100)

    # Calculate Matched Filter Delay for SIVERS
    z_wireless, lags_wireless = delay.matched_filter_calc(
        rx_iq=rx_wireless_upsampled, ref_iq=tx_upsampled, fs=(ofdm_conf.FS * 100)
    )
    z_mag_wireless = np.abs(z_wireless)

    # Find Peak
    peak_idx_wireless = np.argmax(z_mag_wireless)
    fine_delay_wireless = lags_wireless[peak_idx_wireless]

    # Calculate Calibration Constant
    constant = ((fine_delay_wireless) * C) - REFERENCE_DISTANCE
    print(f"Calculated Constant: {constant}")
    return constant


def upsample(raw_data: np.ndarray, scale_factor: int = 100) -> np.ndarray:
    """
    Upsamples raw data based on the scale factor for matched filter delay esiamtion

    Args:
        raw_data: Raw RX or Tx pilot data to be upsampled
        scale_factor: Multiplication factor for upsamples i.ee to upsample from 100Mhz to 10Ghz use scale_factor = 100

    Returns:
        np.ndarray of upsampled data
    """
    N = len(raw_data)
    K = scale_factor
    N_padded = N * K

    # Convert to freq
    freq = np.fft.fftshift(np.fft.fft(raw_data))

    total_zeros = N_padded - N  # Calculate total number of zeros for upsampling
    zeros_side = np.zeros(
        total_zeros // 2
    )  # Get the number of requied zeros to append and prepend to original data

    freq_padded = np.concatenate([zeros_side, freq, zeros_side])
    freq_ready = np.fft.ifftshift(freq_padded)

    upsampled = np.fft.ifft(freq_ready) * K
    return upsampled


def scale_rx_signal(raw_rx_data: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(raw_rx_data))

    if max_val > 0:
        scale_factor = 0.9 / max_val
        scaled_rx_data = raw_rx_data * scale_factor
    return scaled_rx_data


def calculate_precise_delay(rx_signal, ref_signal, fs, upsample_factor=100):
    """
    Calculates delay using coarse correlation zero-padded FFT interpolation
    """
    corr = scipy.signal.correlate(rx_signal, ref_signal, mode="full")
    lags = scipy.signal.correlation_lags(len(rx_signal), len(ref_signal), mode="full")

    # Find coarse peak
    mag = np.abs(corr)
    coarse_idx = np.argmax(mag)

    # Get small window
    radius = 16
    start = max(0, coarse_idx - radius)
    end = min(len(corr), coarse_idx + radius)

    window = corr[start:end]

    # Zero padded interpolation
    window_fft = np.fft.fft(window)

    # Zero pad
    n_original = len(window)
    n_padded = n_original * upsample_factor
    n_zeros = n_padded - n_original

    # FFT shift
    window_fft_shifted = np.fft.fftshift(window_fft)

    # insert zeros
    zeros = np.zeros(n_zeros, dtype=complex)
    fft_padded = np.concatenate(
        [
            window_fft_shifted[: n_original // 2],
            zeros,
            window_fft_shifted[n_original // 2 :],
        ]
    )

    # IFFT
    fft_padded_ready = np.fft.ifftshift(fft_padded)
    window_upsampled = np.fft.ifft(fft_padded_ready) * upsample_factor

    # Find precise peak
    upsampled_mag = np.abs(window_upsampled)
    peak_upsampled_idx = np.argmax(upsampled_mag)

    # Calculate total delay
    fractional_offset = peak_upsampled_idx / upsample_factor

    total_idx = start + fractional_offset

    # Convert to time
    zero_lag_index = np.where(lags == 0)[0][0]
    final_lag_samples = total_idx - zero_lag_index

    return final_lag_samples / fs


if __name__ == "__main__":
    main()

import numpy as np
import json
from ofdm.processing.rx import unpack_rx_file, normalize_rx_signal, extract_packet
from ofdm.channel.delay import calculate_sub_sample_delay_parabolic
from ofdm.modulation import qam
from ofdm.utils.eval import calc_EVM, calc_BER, calc_SER
from ofdm.config import OFDMConfig


def process_dat_file(dat_path: str, ref_path:str, channel:int, ofdm_conf:OFDMConfig) -> dict:
    """
    Processes a single .dat file and returns delay and transfer quality metrics.
    Only the delay is used for localization, demodulated IQ data is discarded
    """

    # get refined_packet_start for delay calculation
    demodulated_data, ref_data, refined_packet_start = unpack_rx_file(
        ofdm_conf = ofdm_conf,
        rx_path = dat_path,
        ref_path = ref_path,
    )

    # re-read raw signal for delay calc — unpack_rx_file applies CFO correction internally
    # which we do not want for the matched filter time delay estimate
    rx_data = normalize_rx_signal(extract_packet(
        rx_data=np.fromfile(dat_path, dtype=np.complex64), 
        start_idx=refined_packet_start, 
        total_symbols = (1 + 1 + ref_data["n_data_symb"]),
        ofdm_conf=ofdm_conf
    ))
    tx_pilot = np.array(ref_data['pilot_ref_real']) + 1j * np.array(ref_data['pilot_ref_imag'])

    delay_s = calculate_sub_sample_delay_parabolic(
        rx_signal=rx_data,
        ref_signal = tx_pilot,
        fs = ofdm_conf.FS
    )
    delay_ns = delay_s * 1e9 # convert to nano seconds

    ref_binary = ref_data['binary_data']
    ref_iq_data = qam.binary_ref_to_iq(binary_string=ref_binary)
    evm = calc_EVM(iq_rx=demodulated_data, iq_ref=ref_iq_data)
    ser = calc_SER(iq_rx=demodulated_data, iq_ref=ref_iq_data)
    ber = calc_BER(iq_rx=demodulated_data, iq_ref = ref_iq_data)

    return {f"delay{channel}":delay_ns,
            f"evm{channel}": evm,
            f"ser{channel}": ser,
            f"ber{channel}": ber}
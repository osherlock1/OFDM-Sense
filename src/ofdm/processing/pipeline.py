import numpy as np
from ofdm.processing.rx import unpack_rx_file


def process_dat_file(dat_path: str, ref_path:str, channel:int, ofdm_conf, usrp_conf) -> dict:
    """
    Helper for getting delays for localization experiemtn.  Note// This is not used for
    getting the actualy data and only the delays currently.
    
    """
    demodulated_data, ref_data, refined_packet_start = unpack_rx_file(
        ofdm_conf = ofdm_conf,
        rx_path = dat_path,
        ref_path = ref_path,
    )
    

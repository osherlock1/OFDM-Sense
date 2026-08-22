import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import os

# Internal
from ofdm.config import OFDMConfig
from ofdm.processing.rx import unpack_rx_file
from ofdm.utils import eval
from ofdm.modulation import qam
from ofdm.utils import usrp

RX_DEFAULT_PATH = "./data_files/rand_ofdm_packet_rx.dat"
TX_DEFAULT_PATH = "./data_files/rand_ofdm_packet.dat"


def main():
    parser = argparse.ArgumentParser(description="Unpack and Plot Recieved OFDM Packet")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="File name of packet to unpack. Defaults to the hardware RX capture, "
        "or the TX file when --sim is set",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="./data_files/rand_ofdm_packet_ref.json",
        help="Reference packet json file name",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Simulate unpacking directly from the generated TX file "
        "(no hardware capture needed)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot Constalation Diagrams of Unpacked OFDM Packets",
    )
    args = parser.parse_args()

    if args.file is None:
        args.file = TX_DEFAULT_PATH if args.sim else RX_DEFAULT_PATH

    # Load Configurations
    USRP_CONFIG_PATH = "./configs/usrp_settings.yaml"
    usrp_conf = usrp.load_config(USRP_CONFIG_PATH)
    ofdm_conf = OFDMConfig()

    rx_channel_idx = usrp_conf.rx_channel_idx.replace(",", "")

    print(f"Unpacking {len(rx_channel_idx)} RX Channels...")

    # Unpack all RX files
    demodulated_dict = {}
    start_idx_list = []

    if (
        not args.sim and args.file == RX_DEFAULT_PATH
    ):  # checks if custom file is inputted in CLA if not runs as normal
        for channel in rx_channel_idx:
            if len(rx_channel_idx) != 1:
                rx_path = "./data_files/rand_ofdm_packet_rx.0" + channel + ".dat"
            print("#####################")
            print("Unpacking Channel" + channel + "...")
            print("#####################\n")
            demodulated_data, ref_data, start_idx = unpack_rx_file(
                ofdm_conf=ofdm_conf, rx_path=rx_path, ref_path=args.ref
            )
            demodulated_dict["Channel_" + channel] = demodulated_data
            start_idx_list.append((channel, start_idx))
    else:
        print("#####################")
        print("Unpacking File" + args.file + "...")
        print("#####################\n")
        demodulated_data, ref_data, start_idx = unpack_rx_file(
            ofdm_conf=ofdm_conf, rx_path=args.file, ref_path=args.ref
        )
        demodulated_dict["Channel"] = demodulated_data
        start_idx_list.append((0, start_idx))

    # --------- Evaluation ---------
    print("#####################")
    print("Calculating OFDM Perforamnce")
    print("#####################")
    # Get Referense Data
    ref_binary = ref_data["binary_data"]
    n_ref_samples = ref_data["n_samples"]
    ref_iq = binary_ref_to_iq(binary_string=ref_binary, n_samples=n_ref_samples)

    # Store Metrics for post processing
    ber_list = []
    ser_list = []
    evm_list = []

    # Eval and plot all unpacked data
    for channel_name, demodulated_data in demodulated_dict.items():
        print(f"Performance metrics for {channel_name}")

        # Calculate EVM
        evm = eval.calc_EVM(iq_rx=demodulated_data, iq_ref=ref_iq)
        print(f"EVM:{evm:.2f}dB")

        # Calculate SER
        ser = eval.calc_SER(iq_rx=demodulated_data, iq_ref=ref_iq)
        print(f"SER:{ser * 100:.2f}%")

        # Calculate BER
        ber = eval.calc_BER(iq_rx=demodulated_data, iq_ref=ref_iq)
        print(f"BER:{ber * 100:.2f}% \n")

        ber_list.append(ber)
        ser_list.append(ser)
        evm_list.append(evm)

    for channel_name, demodulated_data in demodulated_dict.items():
        # ----------- Save Unpacked Data ----------
        unpacked_file_name = f"unpacked_data_{channel_name}.json"
        save_unpacked_data(demodulated_data, file_name=unpacked_file_name)

    if args.plot:
        # ---------- Plots ---------------
        for channel_name, demodulated_data in demodulated_dict.items():
            ref_constalation = qam.get_reference_constalation()

            plt.figure()
            plt.scatter(np.real(demodulated_data), np.imag(demodulated_data), alpha=0.5)
            plt.scatter(np.real(ref_constalation), np.imag(ref_constalation))
            plt.title(f"{channel_name} Constalation plot")
            plt.xlabel("Real")
            plt.ylabel("Imaginary")
        plt.show()

    eval_metrics_path = "./data_files/ofdm_performance.json"
    os.makedirs(os.path.dirname(eval_metrics_path), exist_ok=True)
    json_data = {
        "evm": evm_list,
        "ser": ser_list,
        "ber": ber_list,
        "start_idx": start_idx_list,
    }
    with open(eval_metrics_path, "w") as f:
        json.dump(json_data, f, indent=2, default=float)
    print(f"Stored Eval Metrics to {eval_metrics_path}")


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


def save_unpacked_data(rx_iq: np.ndarray, file_name: str):
    """
    Saves the unpacked data a json file

    Args:
        rx_iq: Unpacked RX iq data (scaled +-3)
        file_name: name of file to save data (do not include data_files/)
    """
    # Convert IQ to Binary
    binary_rx = [qam.iq_to_binary(sample) for sample in rx_iq]

    # Convert rx_iq to list

    unpacked_data = {
        "unpacked_data_real": np.real(rx_iq).tolist(),
        "unpacked_data_imag": np.imag(rx_iq).tolist(),
        "unpacked_binary_data": binary_rx,
    }

    json_path = f"data_files/{file_name}"
    with open(json_path, "w") as f:
        json.dump(unpacked_data, f, indent=2)
    print(f"[Success] Saved Unpacked Data to {json_path}")


if __name__ == "__main__":
    main()

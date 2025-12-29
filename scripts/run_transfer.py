import argparse
import subprocess
import os
import sys
import json
#Internal
from ofdm.utils import usrp


def main():
    parser = argparse.ArgumentParser(description="Run USRP Transfer via C++ Driver")
    #File I/O
    parser.add_argument("--tx_file", type=str, default = "rand_ofdm_packet.dat", help="File USRP will read from")
    parser.add_argument("--rx_file", type=str, default = "rand_ofdm_packet_rx.dat", help="File USRP will write to")

    #Hardware Configuration
    parser.add_argument("--channel", "-c", type=str, required=True, help="Channel Selection (A = Baseband, B = Frontend)")
    parser.add_argument("--tx_addr", type=str, default="192.168.30.2", help ="IP of TX")
    parser.add_argument('--rx_addr', type=str, default ="192.168.30.2", help ='IP of RX')
    parser.add_argument('--rate', type=float, default = 100e6, help="Samlpe Rate(Hz)")
    parser.add_argument('--freq', type=float, default=60e6, help="Carrier frequency (Hz)")
    parser.add_argument('--gain', type=float, default=0, help = "TX/RX Gain (dB)")

    # C++ Binary Path
    parser.add_argument('--bin', type=str, default ="./build/TXRX_FROM_FILE", help = "Path to C++ USRP Driver executable")

    args= parser.parse_args()

    #Validation
    if not os.path.exists(f"data_files/{args.tx_file}"):
        print(f"[Error] TX file not found: {args.tx_file}")
        sys.exit(1)
    if not os.path.exists(f"data_files/{args.bin}"):
        print(f"[Error] C++ Executable not foudn:{args.bin}")
        print("Make sure to run 'cmake .. && make")
    
    #Configure Hardware
    config = usrp.USRPConfig(
        build_path = args.bin,
        tx_addr=f"addr={args.tx_addr}",
        rx_addr=f"addr={args.rx_addr}",
        tx_rate=args.rate,
        rx_rate=args.rate,
        tx_freq=args.freq,
        rx_freq=args.freq,
        tx_gain=args.gain,
        rx_gain=args.gain
    )

    #Get number of samlpes to transfer
    ref_path = f"data_files/rand_ofdm_packet_ref.json"
    with open(ref_path, "r") as f:
        data = json.load(f)
        n_samps = data['n_samples']
    
    n_tx_samps = n_samps
    n_rx_samlpes = n_samps + 2000 # Add buffer

    print(f"Transferring {n_tx_samps} samlpes (expecting {n_rx_samlpes} RX)...")

    #Execute the URSP Transfer
    usrp.run_transfer(
        config=config,
        tx_file=f"data_files/{args.tx_file}",
        rx_file=f"data_files/{args.rx_file}",
        nsamps=n_rx_samlpes,
        channel=args.channel
    )
    

    
if __name__ == "__main__":
    main()
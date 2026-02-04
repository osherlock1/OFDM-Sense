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
    parser.add_argument("--tx_file", type=str, default = "data_files/rand_ofdm_packet.dat", help="File USRP will read from")
    parser.add_argument("--rx_file", type=str, default = "data_files/rand_ofdm_packet_rx.dat", help="File USRP will write to")
    parser.add_argument("--ref_path", type=str, default = "data_files/rand_ofdm_packet_ref.json", help = "Path to reference json file")
    #Hardware Configuration
    parser.add_argument("--channel", "-c", type=str, help="subdev specification: (A:0 B:0) default")
    parser.add_argument("--tx_addr", type=str, help ="IP of TX")
    parser.add_argument('--rx_addr', type=str, help ='IP of RX')
    parser.add_argument('--rate', type=float, help="Samlpe Rate(Hz)")
    parser.add_argument('--freq', type=float, help="Carrier frequency (Hz)")
    parser.add_argument('--gain', type=float, help = "TX/RX Gain (dB)")
    parser.add_argument('--tx_channel_idx', type=str, help="Specify the channel to send from (based on subdev specification)")
    parser.add_argument('--rx_channel_idx', type=str, help="Specify the channel to recieve from (based on subdev specification)")

    # C++ Binary Path
    parser.add_argument('--bin', type=str, help = "Path to C++ USRP Driver executable")

    args= parser.parse_args()

    #Validation
    if not os.path.exists(args.tx_file):
        print(f"[Error] TX file not found: {args.tx_file}")
        sys.exit(1)

    
    #Load default config from config file
    DEFAULT_CONFIG_PTH = "./configs/usrp_settings.yaml"
    usrp_conf = usrp.load_config(path=DEFAULT_CONFIG_PTH)

    #Overwrite configs with CLI args
    if args.channel is not None:
        usrp_conf.subdev = args.channel
        print(f"[USRP CONFIG] Updated subdev to:{args.channel}")
    if args.tx_addr is not None:
        usrp_conf.tx_addr = args.tx_addr
        print(f"[USRP CONFIG] Updated tx_addr to:{args.tx_addr}")
    if args.rx_addr is not None:
        usrp_conf.rx_addr = args.tx_addr
        print(f"[USRP CONFIG] Updated rx_addr to:{args.rx_addr}")
    if args.rate is not None:
        usrp_conf.tx_rate = args.rate
        usrp_conf.rx_rate = args.rate
        print(f"[USRP CONFIG] Updated rate to:{args.tx_rate}")
    if args.freq is not None:
        usrp_conf.rx_freq = args.freq
        usrp_conf.tx_freq = args.freq
        print(f"[USRP CONFIG] Updated freq to {args.freq}")
    if args.gain is not None:
        usrp_conf.rx_gain = args.gain
        usrp_conf.tx_gain = args.gain
        print(f"[USRP CONFIG] Updated gain to{args.gain}")
    if args.tx_channel_idx is not None:
        usrp_conf.tx_channel_idx = args.tx_channel_idx
        print(f"[USRP CONFIG] Updated tx_channel_idx to:{args.tx_channel_idx}")
    if args.rx_channel_idx is not None:
        usrp_conf.rx_channel_idx = args.rx_channel_idx
        print(f"[USRP Config] Updated rx_channel_idx to:{args.rx_channel_idx}")
    if args.bin is not None:
        usrp_conf.build_path = args.bin
        print(f"[USRP CONFIG] Updated build path to:{args.bin}")
        
    #Validate Build Path
    if not os.path.exists(usrp_conf.build_path):
        print(f"[Error] C++ Executable not foudn:{usrp_conf.build_path}")
        print("Make sure to run 'cmake .. && make")
        sys.exit(1)
        

    #Get number of samlpes to transfer
    ref_path = args.ref_path
    with open(ref_path, "r") as f:
        data = json.load(f)
        n_samps = data['n_samples']
    
    n_tx_samps = n_samps
    n_rx_samlpes = n_samps + 2000 # Add buffer

    print(f"Transferring {n_tx_samps} samlpes (expecting {n_rx_samlpes} RX)...")

    #Execute the URSP Transfer
    usrp.run_transfer(
        config=usrp_conf,
        tx_file=args.tx_file,
        rx_file=args.rx_file,
        nsamps=n_rx_samlpes,
    )
    

if __name__ == "__main__":
    main()
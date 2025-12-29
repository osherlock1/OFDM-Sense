import numpy as np
import argparse

#Internal
from ofdm.utils import generator
from ofdm.config import OFDMConfig
from ofdm.core import payload, preamble
from ofdm.channel import noise

def main():
    parser = argparse.ArgumentParser(description="Generate OFDM Packet")
    parser.add_argument('--n_symb', '-n', type=int, default=5, help="Number of Payload Symbols.")
    parser.add_argument('--snr', type=float, default=100.0, help="Add SNR in dB")
    parser.add_argument('--seed', type=int, default=42, help = "Random seed")
    args = parser.parse_args()

    #Initialize Configuration
    ofdm_conf = OFDMConfig()
    ofdm_gen = generator.DataGenerator()

    print(f"[Generator] Creating packet: {args.n_symb} symbols, SNR={args.snr}dB")

    #Generate the preamble
    sync_freq = preamble.generate_sync_symbol(config=ofdm_conf, seed=args.seed)
    pilot_freq = preamble.generate_pilot_symbol(config=OFDMConfig, seed=args.seed)
    


if __name__ == "__main__":
    main()


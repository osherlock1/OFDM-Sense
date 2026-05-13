import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import json
import pathlib
#Internal
from ofdm.utils import generator
from ofdm.config import OFDMConfig
from ofdm.core import payload, preamble, waveform
from ofdm.channel import noise
from ofdm.viz import plotter

def main():
    parser = argparse.ArgumentParser(description="Generate OFDM Packet")
    parser.add_argument('--n_symb', '-n', type=int, default=5, help="Number of Payload Symbols.")
    parser.add_argument('--snr', type=float, default=100.0, help="Add SNR in dB")
    parser.add_argument('--seed', type=int, default=42, help = "Random seed")
    parser.add_argument('--out', type=str, default="rand_ofdm_packet", help ="Name of Output File")
    args = parser.parse_args()

    #Initialize Configuration
    ofdm_conf = OFDMConfig()
    ofdm_gen = generator.DataGenerator(config=ofdm_conf)

    print(f"[Generator] Creating packet: {args.n_symb} symbols, SNR={args.snr}dB")

    #Generate the preamble
    sync_freq = preamble.generate_sync_symbol(config=ofdm_conf, seed=args.seed)
    sync_final = waveform.create_time_domain_symbol(sync_freq, ofdm_conf.CP_LEN)

    pilot_freq = preamble.generate_pilot_symbol(config=ofdm_conf, seed=args.seed)
    pilot_final = waveform.create_time_domain_symbol(pilot_freq, ofdm_conf.CP_LEN) 

    #Start the final generated packet
    packet_time_symbols = [sync_final, pilot_final]

    #Build the payload
    binary_ref_list = []

    for i in range(args.n_symb):
        #Generate Random Data
        iq_data, bits_str = ofdm_gen.generate_random_payload()
        binary_ref_list.append(bits_str)

        #Generate Pilots
        pilot_vals = preamble.generate_pilot_vals(config = ofdm_conf, seed=args.seed)

        #Map to Symbol
        data_symbol_freq = payload.generate_ofdm_data_symbol(config=ofdm_conf, iq_data = iq_data, pilot_data=pilot_vals)

        #Convert to time domain
        data_symbol_final = waveform.create_time_domain_symbol(data_symbol_freq, cp_len = ofdm_conf.CP_LEN)
        
        #Append to the final signal
        packet_time_symbols.append(data_symbol_final)
    
    #Concatenate into final packet
    full_packet = np.concatenate(packet_time_symbols)

    #Add noise
    if args.snr < 100.0:
        full_packet = noise.add_noise(signal = full_packet, snr_db=args.snr, seed=args.seed)
    
    #Add buffer prefix
    buffer = np.zeros(20000, dtype=complex)
    final_tx_signal = np.concatenate([buffer, full_packet, buffer])

    #Scale Signal between +-0.9
    max_val = np.max(np.abs(final_tx_signal))

    if max_val > 0:
        scale_factor = 0.9 / max_val
        final_tx_signal = final_tx_signal * scale_factor
    


    #------Save Data to File-------
    
    #Create data dir if it does not exist
    os.makedirs("data_files", exist_ok=True)

    #Save Binary (.dat) for USRP
    bin_path = f"data_files/{args.out}.dat"
    final_tx_signal.astype(np.complex64).tofile(bin_path)
    print(f"[Success] Saved bianry data to {bin_path}")

    #Save JSON referense for python post-processing
    referense_data = {
        "seed":args.seed,
        "n_data_symb":args.n_symb,
        "n_samples": len(final_tx_signal),
        "binary_data": binary_ref_list,
        "samples_real": np.real(final_tx_signal).tolist(),
        "samples_imag": np.imag(final_tx_signal).tolist(),
        "sync_ref_real": np.real(sync_final).tolist(),
        "sync_ref_imag": np.imag(sync_final).tolist(),
        "pilot_ref_real": np.real(pilot_final).tolist(),
        "pilot_ref_imag": np.imag(pilot_final).tolist() 
    }

    json_path = f"data_files/{args.out}_ref.json"
    with open(json_path, "w") as f:
        json.dump(referense_data, f ,indent=2)
    print(f"[Success] Saved Referense Data to {json_path}")

    #Debugg Plotting
    # plotter.plot_time_series(final_tx_signal, title="Final TX Signal")
    # plt.show()


if __name__ == "__main__":
    main()


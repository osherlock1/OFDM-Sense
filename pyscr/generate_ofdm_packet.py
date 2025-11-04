import argparse 
from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
from data_generator import DataGenerator
import numpy as np
from sync_symbol import SyncSymbol
import matplotlib.pyplot as plt
import json
from pilot_symbol import PilotSymbol

def main():
    #Create parser
    parser = argparse.ArgumentParser()

    #Add arguments
    parser.add_argument('--data_source', type=str, default = "random",
                       help = "Specify which file the data is stored in, random for random data: Default = Random")
    parser.add_argument('--snr', type = int, default = 100)
    parser.add_argument('--seed', type = int, default = None)
    #Parse arguments
    args = parser.parse_args()
    seed = args.seed
    snr = args.snr
    if args.data_source == "random":
        gen_random_packet(5, seed = seed, snr_db=snr)
        print(f"Generated OFDM packet with seed = {seed}, SNR = {snr}")
    

def gen_random_packet(N_data_symb: int = 5, file_name:str = "rand_ofdm_packet", seed = None, snr_db:int = 100) -> np.ndarray:
    
    map = SubcarrierMap()
    om = OFDMManager()
    if seed is not None: 
        dg = DataGenerator(seed = seed)
    else:
        dg = DataGenerator()

    # --------- FIX LATER -----------------
    #for now just hard hard coded pilots
    pilots = np.array([1, 1, 1, 1], dtype=complex)


    ofdm_data_symbols = []

    print(f"Generating Random OFDM Packet with {N_data_symb} Data Packets...")
    #Generate random binary data
    rand_data_ref = []
    for i in range(N_data_symb):
        #Keep track of random data
        rand_data = dg.generate_random_binary(len(map.data_bins) * 4)
        rand_data_ref.append(rand_data)

        parsed_input = dg._parse_string(rand_data, 4)

        #Convert Binary -> IQ
        iq_input = []
        for word in parsed_input:
            iq_input.append(om.binary_to_iq(word))

        #Build the OFDM symbol and add to packet list
        input_array = np.array(iq_input, dtype = complex)
        ofdm_symbol = OFDMSymbol(iq_samples48=input_array, pilots4=pilots)
        ofdm_tx = om.create_tx_block(ofdm_symbol)
        ofdm_data_symbols.append(ofdm_tx)
    

    #Develop final packet to be returned
    final_packet = om.create_tx_block(SyncSymbol())
    final_packet = np.concatenate([final_packet, om.create_tx_block(PilotSymbol())])
    
    for symbol in ofdm_data_symbols:
        final_packet = np.concatenate([final_packet, symbol])

    if snr_db != 100:
        final_packet = dg.add_noise(final_packet, snr_db)

    buffer = np.zeros(30, dtype=complex)
    final_packet = np.concatenate([buffer, final_packet, buffer])
    
    print(f"Generation Copmlete! \n")

    #Save final packet to a file
    data_file = "data_files/"
    save_path = data_file + file_name + ".dat"
    final_packet.astype(np.complex64).tofile(save_path)
    print(f"Saved OFDM Packet to {save_path}!")
    #Save binary reference
    referense_data = {
        "seed":seed,
        "N_data_symbols": N_data_symb,
        "binary_data:" : rand_data_ref,
    }    
    json_path = data_file + file_name + "_ref.json"
    
    with open(json_path, "w") as f:
        json.dump(referense_data, f, indent = 2)
    print(f"Saved Binary Reference to {json_path}!\n")
    





if __name__ == "__main__":
    main()
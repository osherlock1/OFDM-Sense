import numpy as np
import json
import matplotlib.pyplot as plt
#internal
from ofdm.config import OFDMConfig
from ofdm.modulation import qam
from ofdm.utils import eval

def main():
    ofdm_conf = OFDMConfig()

    #Get RX Data
    rx_data_file_name = "unpacked_data.json"
    rx_data = unpack_json(rx_data_file_name)
    
    #Get Ref Data
    ref_data_file_name = "rand_ofdm_packet_ref.json"
    ref_data = unpack_json(ref_data_file_name)

    #Unpack Json Files
    rx_iq = np.array(rx_data["unpacked_data_real"]) + 1j * np.array(rx_data["unpacked_data_imag"])
    rx_binary = rx_data["unpacked_binary_data"]
    
    #Unpack Ref Data
    ref_binary_string = ref_data['binary_data']
    n_samples = ref_data['n_samples']
    n_sym = ref_data['n_data_symb']
    ref_iq = binary_ref_to_iq(binary_string=ref_binary_string, n_samples=n_samples)
    
    sc_metrics = per_subcarrier_eval(rx_iq_data = rx_iq, ref_iq_data = ref_iq, config=ofdm_conf, n_sym = n_sym)

def per_subcarrier_eval(rx_iq_data:np.ndarray, ref_iq_data:np.ndarray, config:OFDMConfig, n_sym:int):
    """  
    Evaluate the Performance of Individual Subcarriers
    """

    n_data_carriers = len(config.data_carriers)

    rx_grid = rx_iq_data.reshape(-1, n_data_carriers)
    ref_grid = ref_iq_data.reshape(-1, n_data_carriers)
    
    sc_metrics = {
        "indices":[],
        "evm_db": []
    }

    print(f"Evaluating {n_data_carriers} subcarriers...")

    for col_idx, sc_index in enumerate(config.data_carriers):
        #Extract the COlUMN
        rx_col = rx_grid[:, col_idx]
        ref_col = ref_grid[:, col_idx]

        #Calculate Metric for this subcarrier
        evm = eval.calc_EVM(iq_rx=rx_col, iq_ref=ref_col)

        #Store Results
        sc_metrics['indices'].append(sc_index)
        sc_metrics['evm_db'].append(evm)
    
    
    k = sc_metrics['indices']
    evm = sc_metrics['evm_db']

    db_threshold = -4

    for i, metric in enumerate(evm):
        if metric >= db_threshold:
            print(f"Subcarrier{k[i]}, EVM:{metric:.2f}dB")
    plt.figure()
    plt.stem(k, evm)
    plt.title("EVM per subcarrier")
    plt.show()

    


    
    

    

def unpack_json(json_file_name:str)->dict:
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

def binary_ref_to_iq(binary_string:str, n_samples:int)->np.ndarray:

    full_string = "".join(binary_string)

    #Parse String into 4 bit words
    word_len = 4
    binary_word_list = np.array([full_string[i:word_len + i] for i in range(0 ,len(full_string), word_len)])
    
    #Convert to IQ
    iq_array = [qam.binary_to_iq(word) for word in binary_word_list]
    return np.array(iq_array) * np.sqrt(10)





if __name__ == "__main__":
    main()
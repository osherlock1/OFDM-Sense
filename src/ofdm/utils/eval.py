import numpy as np
from typing import Tuple
#Internal
from ofdm.modulation import qam

def calc_EVM(iq_rx:np.ndarray, iq_ref:np.ndarray)->float:
    """  
    Calculated the Error Vector Magnitude

    Args:
        iq_rx: The unpacked RX samples
        rq_ref: The reference IQ samlpes (from json ref)

    Returns:
        evm: Error Vector Magnitude
    """
    
    #Length Check
    if len(iq_rx) != len(iq_ref):
        print(f"[EVM] RX Data:{len(iq_rx)} samples does not match Reference Lengt:{len(iq_ref)}...")
        
        min_len = min(len(iq_ref), len(iq_rx))
        iq_rx = iq_rx[:min_len]
        iq_ref = iq_ref[:min_len]
        print(f"[EVM] Truncating RX and Referense to {min_len} samples.")
    
    N = len(iq_rx)
    sum_result = 0

    for i in range(N):
        Ierr = np.real(iq_rx[i] - np.real(iq_ref[i]))
        Qerr = np.imag(iq_rx[i]) - np.imag(iq_ref[i])
        sum_result += ((Ierr ** 2) + (Qerr ** 2)) / (np.abs(iq_ref[i]) ** 2)
    
    evm = np.sqrt((1 / N) * sum_result)

    #Convert to dB
    return 20 * np.log10(evm)

def calc_SER(iq_rx:np.ndarray, iq_ref:np.ndarray)->float:
    """  
    Calculates Symbol Error Rate

    Args:
        iq_rx: Recieved IQ data (+-3 Scale)
        iq_ref: Reference IQ data (+-3 Scale)

    Returns:
        ser: Symbol Error Rate
    """

    #Convert rx iq to Binary
    binary_rx = [qam.iq_to_binary(sample) for sample in iq_rx]

    #Convert ref iq to Binary
    binary_ref = [qam.iq_to_binary(sample) for sample in iq_ref]

    #Length Check
    if len(binary_rx) != binary_ref:
        print(f"[SER] RX len:{len(binary_rx)} samples does not match referense length:{len(binary_ref)} samples")
        min_len = min(len(binary_ref), len(binary_rx))
        print(f"Truncating to min length:{min_len} samples")
        binary_rx = binary_rx[:min_len]
        binary_ref = binary_ref[:min_len]
    
    n_samples = len(binary_ref)

    #Calculate SER
    Errors = np.sum(binary_rx[i] != binary_ref[i] for i in range(n_samples))

    ser = Errors / n_samples

    return ser

def calc_BER(iq_rx:np.ndarray, iq_ref:np.ndarray)->float:
    """
    Calculate Bit Error Rate (BER) by comparing the difference between individual bits

    Args:
        iq_rx = Recieved IQ samples (-+ 3 Scaled)
        iq_ref = Reference IQ samples (+-3 Scaled)

    Returns:
        BER = Bit Error Rate
    """

    #Convert iq_rx to binary
    binary_rx = [qam.iq_to_binary(sample) for sample in iq_rx]

    #Convert iq_ref to binary
    binary_ref = [qam.iq_to_binary(sample) for sample in iq_ref]

    #Convert into continous string
    binary_string_rx = "".join(binary_rx)
    binary_string_ref = "".join(binary_ref)

    #Check Len
    if len(binary_string_ref) != len(binary_string_rx):
        print(f"[BER] Lenth of RX:{len(binary_string_rx)} does not equal Length of Ref:{len(binary_string_ref)}")
        min_len = min(len(binary_string_ref), len(binary_string_rx))
        print(f"Truncating to {min_len} samples")
        binary_string_rx = binary_string_rx[:min_len]
        binary_string_ref = binary_string_ref[:min_len]
    
    n_bits = len(binary_string_rx)
    bit_errors = np.sum(binary_string_ref[i] != binary_string_rx[i] for i in range(n_bits))

    #Calc BER
    ber = bit_errors / n_bits

    return ber
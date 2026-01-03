import numpy as np


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

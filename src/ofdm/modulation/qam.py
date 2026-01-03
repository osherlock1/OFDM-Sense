import numpy as np
from typing import Dict, Tuple, List


QAM16_MAP: Dict[str, complex] = {
    "0000": complex(-3, -3),
    "0001": complex(-3, -1),
    "0010": complex(-3,  3),
    "0011": complex(-3,  1),
    "0100": complex(-1, -3),
    "0101": complex(-1, -1),
    "0110": complex(-1,  3),
    "0111": complex(-1,  1),
    "1000": complex( 3, -3),
    "1001": complex( 3, -1),
    "1010": complex( 3,  3),
    "1011": complex( 3,  1),
    "1100": complex( 1, -3),
    "1101": complex( 1, -1),
    "1110": complex( 1,  3),
    "1111": complex( 1,  1),
}

DEFAULT_SCALE = np.sqrt(10)

def binary_to_iq(binary_string: str, scale: float = DEFAULT_SCALE) -> complex:
    """  
    Converts a 4-bit binary string to a complex IQ sample from the QAM 16 mapping
    """

    if len(binary_string) != 4:
        raise ValueError(f"Length of input binary: {len(binary_string)} is not 4 bits.")
    
    if binary_string not in QAM16_MAP:
        raise ValueError(f"Invalid Binary string: {binary_string}")
    
    val = QAM16_MAP[binary_string]
    return val / scale

def iq_to_binary(iq_sample: complex, scale:float = DEFAULT_SCALE) -> str:
    """  
    Takes an IQ sample and converts it into a binary string mapping of QAM 16
    """

    sample_rescaled = iq_sample# * scale

    min_dist = float('inf')
    closest_bits = "0000"

    #Seach for closes qam16 mapping
    for bits, ref_point in QAM16_MAP.items():
        # Calculate distance
        dist = abs(sample_rescaled - ref_point)

        if dist < min_dist:
            min_dist = dist
            closest_bits = bits
    
    return closest_bits

def get_reference_constalation(scale:float = DEFAULT_SCALE) -> np.ndarray:
    """  
    Returns the reference QAM constalation for easy plotting
    """

    points = list(QAM16_MAP.values())
    return np.array(points) / scale


    
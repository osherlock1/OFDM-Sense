import numpy as np
from typing import Dict, Tuple


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
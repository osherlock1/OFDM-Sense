from dataclasses import dataclass

@dataclass
class OFDMConfig:
    N: int = 64
    CP_LEN: int = 16
    FS: float = 100e6
    DATA_CARRIERs: list = None 
    PILOT_CARRIERs: list = None

    def __post_init__(self):
        # logical initialization
        pass

    
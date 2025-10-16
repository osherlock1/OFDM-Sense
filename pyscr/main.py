import matplotlib.pyplot as plt
import numpy as np
from subcarrier_map import SubcarrierMap
from ofdm_symbol import OFDMSymbol
from ofdm_manager import OFDMManager


    

def main():


    map = SubcarrierMap()
    om = OFDMManager(map)

     #Simple IQ samples respoinse
    iq_samples = np.array([-0.9487 + 0.9487j, -0.3162 + 0.9487j,  0.3162 + 0.9487j,  0.9487 + 0.9487j,
                            -0.9487 + 0.3162j, -0.3162 + 0.3162j,  0.3162 + 0.3162j,  0.9487 + 0.3162j,
                            -0.9487 - 0.3162j, -0.3162 - 0.3162j,  0.3162 - 0.3162j,  0.9487 - 0.3162j,
                            -0.9487 - 0.9487j, -0.3162 - 0.9487j,  0.3162 - 0.9487j,  0.9487 - 0.9487j,

                            -0.9487 + 0.9487j, -0.3162 + 0.9487j,  0.3162 + 0.9487j,  0.9487 + 0.9487j,
                            -0.9487 + 0.3162j, -0.3162 + 0.3162j,  0.3162 + 0.3162j,  0.9487 + 0.3162j,
                            -0.9487 - 0.3162j, -0.3162 - 0.3162j,  0.3162 - 0.3162j,  0.9487 - 0.3162j,
                            -0.9487 - 0.9487j, -0.3162 - 0.9487j,  0.3162 - 0.9487j,  0.9487 - 0.9487j,

                            -0.9487 + 0.9487j, -0.3162 + 0.9487j,  0.3162 + 0.9487j,  0.9487 + 0.9487j,
                            -0.9487 + 0.3162j, -0.3162 + 0.3162j,  0.3162 + 0.3162j,  0.9487 + 0.3162j,
                            -0.9487 - 0.3162j, -0.3162 - 0.3162j,  0.3162 - 0.3162j,  0.9487 - 0.3162j,
                            -0.9487 - 0.9487j, -0.3162 - 0.9487j,  0.3162 - 0.9487j,  0.9487 - 0.9487j], dtype=complex)


    pilots = np.array([1 + 0j, 1 + 0j, 1+0j, 1+0j], dtype=complex)

    ofdm_symbol1 = OFDMSymbol(iq_samples48=iq_samples, pilots4=pilots, submap=map)
    ofdm_symbol2 = OFDMSymbol(iq_samples48=iq_samples, pilots4=pilots, submap=map)
    TX_Block2 = om.create_tx_block(ofdm_symbol1)
    

    om.ifft(ofdm_symbol2)
    TX_Block = om.add_cycle_prefix(ofdm_symbol2)


    plt.figure()
    plt.plot(TX_Block)
    plt.title("TX_BLOCK")
    plt.figure()
    plt.plot(TX_Block2)
    plt.title("TX_BLOCK2")
    plt.show()

if __name__ == "__main__":
    main()



import matplotlib.pyplot as plt
import numpy as np
from subcarrier_map import SubcarrierMap
from ofdm_symbol import OFDMSymbol
from ofdm_manager import OFDMManager
#Indicies helper




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
    
    print(ofdm_symbol1.symbol)
    
    for i in (list(range(-32,32))):
        print(ofdm_symbol1.symbol[map.idx(i)])

    om.ifft(ofdm_symbol1)
    TX_Block = om.add_cycle_prefix(ofdm_symbol1)



    print(ofdm_symbol1.symbol)
    plt.plot(TX_Block)
    plt.show()

if __name__ == "__main__":
    main()



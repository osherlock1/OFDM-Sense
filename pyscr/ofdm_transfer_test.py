import numpy as np
from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
import matplotlib.pyplot as plt
from data_generator import DataGenerator

def main():
    
    #Instantaite objects
    map = SubcarrierMap()
    om = OFDMManager()
    data_gen = DataGenerator()

    #Genearte random OFDM pakcet
    packet = data_gen.generate_random_packet()

    plt.figure()
    plt.plot(packet)
    plt.show()



if __name__ == "__main__":
    main()

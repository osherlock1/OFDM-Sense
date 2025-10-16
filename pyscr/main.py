from ofdm import ofdmManager
import matplotlib.pyplot as plt
import numpy as np


#Indicies helper
def idx(k):
    return k % 64



def main():

     #Simple IQ samples respoinse
    iq_samples = [-0.9487 + 0.9487j, -0.3162 + 0.9487j,  0.3162 + 0.9487j,  0.9487 + 0.9487j,
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
 -0.9487 - 0.9487j, -0.3162 - 0.9487j,  0.3162 - 0.9487j,  0.9487 - 0.9487j]


    om = ofdmManager()

    #Define the OFDM symbol
    X = np.zeros(64, dtype=complex)

    used_neg = list(range(-26,0))
    used_pos = list(range(1,27))
    pilots_k = [-21,-7,7,21]

    data_bins = []
    
    for k in (used_neg + used_pos):
        if k not in pilots_k:
            data_bins.append(k)


    #Fill in X with the data samples
    i = 0
    for k in data_bins:
        X[idx(k)] = iq_samples[i]
        i += 1

    print(X)
   

    #Perform 64 Poinrt IFFT
    ifft_response = om.ifft(iq_samples, N=64)
    

    #Normalize

    #plt.plot(np.real(iq_samples), np.imag(iq_samples), ".")
    # plt.plot(np.abs(ifft_response))
    # plt.show()


if __name__ == "__main__":
    main()


